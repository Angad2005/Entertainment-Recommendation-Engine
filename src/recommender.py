try:
    import faiss
except Exception:
    faiss = None  # Fallback will be used if faiss import fails
import numpy as np
import torch
import polars as pl
import os
from model import get_device

class Recommender:
    def __init__(self, model, item_df, embedding_dim=128):
        self.model = model
        self.item_df = item_df
        self.embedding_dim = embedding_dim
        self.device = get_device()
        self.index = None
        self.item_embeddings = None
        self.use_faiss = faiss is not None

    def build_index(self, item_features, max_items: int = 20000, cache_dir: str = "data"):
        """Build an index for item embeddings with optional on‑disk caching.

        * Uses FAISS if available; otherwise falls back to a NumPy linear scan.
        * Optionally limits the index to the ``max_items`` most popular titles
          (by ``numVotes``) to keep memory and build time reasonable.
        * Caches the computed embeddings and filtered index mapping to ``cache_dir``
          so subsequent runs can load them instantly.
        """
        # Determine cache file paths for embeddings, filtered indices, and optional FAISS index
        embed_path = os.path.join(cache_dir, "item_embeddings.npy")
        idx_path = os.path.join(cache_dir, "filtered_indices.npy")
        faiss_path = os.path.join(cache_dir, "faiss_index.bin")

        # If cached files exist, load them to avoid recomputation
        if os.path.exists(embed_path) and os.path.exists(idx_path):
            self.item_embeddings = np.load(embed_path)
            self.filtered_indices = np.load(idx_path).tolist()
            # If a persisted FAISS index exists, load it (only when FAISS is available)
            if self.use_faiss and os.path.exists(faiss_path):
                try:
                    self.index = faiss.read_index(faiss_path)
                except Exception:
                    self.index = None
        else:
            # Optionally filter to most popular items for scalability
            if max_items and self.item_df.shape[0] > max_items:
                top_df = self.item_df.sort('numVotes', descending=True).head(max_items)
                top_tconsts = set(top_df['tconst'].to_list())
                mask = self.item_df['tconst'].is_in(top_tconsts)
                self.filtered_indices = [i for i, m in enumerate(mask) if m]
                filtered_features = item_features[self.filtered_indices]
            else:
                self.filtered_indices = list(range(self.item_df.shape[0]))
                filtered_features = item_features

            # Compute embeddings for the (potentially) filtered feature set
            self.model.to(self.device)
            self.model.eval()
            item_tensor = torch.FloatTensor(filtered_features).to(self.device)
            with torch.no_grad():
                self.item_embeddings = self.model.get_item_embedding(item_tensor).cpu().numpy()
            # Persist embeddings for future runs
            os.makedirs(cache_dir, exist_ok=True)
            np.save(embed_path, self.item_embeddings)
            np.save(idx_path, np.array(self.filtered_indices, dtype=np.int64))

        # Compute max number of votes for popularity scaling
        self.max_votes = int(self.item_df['numVotes'].max()) if 'numVotes' in self.item_df.columns else 1

        d = self.embedding_dim
        if self.use_faiss:
            # Use CPU-only FAISS index to avoid GPU initialization issues on WSL
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.item_embeddings.astype('float32'))
            # Persist the FAISS index for future runs (optional, ignore failures)
            try:
                faiss.write_index(self.index, os.path.join(cache_dir, "faiss_index.bin"))
            except Exception:
                pass
        else:
            # No FAISS: keep embeddings for fallback linear scan
            self.index = None


    def _brute_force_search(self, user_embedding, top_k, candidate_factor=5):
        """Fallback search using NumPy when FAISS is unavailable.
        Returns distances and indices analogous to FAISS output.
        """
        # Use the stored item embeddings (already filtered if applicable)
        embeddings = self.item_embeddings
        # Compute L2 distance to each candidate
        diffs = embeddings - user_embedding
        distances = np.linalg.norm(diffs, axis=1)
        # Return the nearest ``top_k * candidate_factor`` items
        sorted_idx = np.argsort(distances)[: top_k * candidate_factor]
        return distances[sorted_idx], sorted_idx

    def recommend(self, user_features, exclude_tconsts=None, alpha=0.7, beta=0.3, top_k=20):
        """Generate top‑k recommendations, excluding already seen titles.
        Supports both FAISS‑based and pure‑NumPy similarity.
        """
        user_features_tensor = torch.FloatTensor(user_features).to(self.device)
        self.model.eval()
        with torch.no_grad():
            user_embedding = self.model.get_user_embedding(user_features_tensor).cpu().numpy().astype('float32')

        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(user_embedding, top_k * 5)
        else:
            distances, indices = self._brute_force_search(user_embedding, top_k)

        results = []
        exclude_set = set(exclude_tconsts) if exclude_tconsts else set()
        for dist, idx in zip(distances, indices):
            # Map filtered index back to original DataFrame row if filtering was applied
            if hasattr(self, 'filtered_indices'):
                # idx refers to position in filtered_embeddings
                original_idx = self.filtered_indices[idx]
            else:
                original_idx = idx
            if original_idx >= len(self.item_df):
                continue
            item = self.item_df.row(original_idx, named=True)
            if item["tconst"] in exclude_set:
                continue
            sim_score = 1 / (1 + dist)
            # Use popularity (numVotes) instead of raw rating for a more famous‑item bias
            popularity = item.get("numVotes", 0) / self.max_votes if getattr(self, 'max_votes', 1) else 0
            hybrid_score = alpha * sim_score + beta * popularity
            results.append({
                "tconst": item["tconst"],
                "primaryTitle": item["primaryTitle"],
                "score": hybrid_score,
                "averageRating": item["averageRating"]
            })
            if len(results) >= top_k:
                break
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results

def prepare_item_features(df, all_genres):
    """
    Convert Polars DF to feature matrix.
    """
    # Multi-hot genres
    genre_data = []
    for genres_str in df["genres"].to_list():
        if genres_str is None:
            genre_data.append([0] * len(all_genres))
            continue
        gs = set(genres_str.split(","))
        genre_data.append([1 if g in gs else 0 for g in all_genres])
    
    # Other features (normalized)
    # Runtime, StartYear
    # Handle nulls
    runtime = df["runtimeMinutes"].fill_null(0).to_numpy() / 300.0
    year = (df["startYear"].fill_null(2000).to_numpy() - 1900) / 150.0
    
    features = np.hstack([
        np.array(genre_data),
        runtime.reshape(-1, 1),
        year.reshape(-1, 1)
    ])
    return features.astype('float32')

def prepare_user_features(user_history, all_titles_df, all_genres, user_selected_genres=None, dm=None):
    """
    Build user feature vector from history and explicitly selected genres.
    """
    user_selected_genres = user_selected_genres or []
    genre_counts = np.zeros(len(all_genres))
    
    # Base weights from explicitly selected genres
    for g in user_selected_genres:
        if g in all_genres:
            genre_counts[all_genres.index(g)] = 0.5 # Give base weight to selected genres

    if not user_history:
        # Default/Cold start features
        if np.sum(genre_counts) > 0:
            genre_counts /= np.sum(genre_counts)
        features = np.hstack([genre_counts, np.array([0.0, 0.0])])
        return features.astype('float32').reshape(1, -1)
    
    # Aggregate genres of liked items
    liked_tconsts = [t for t, v in user_history.items() if v.get('preference') == 'like']
    
    if liked_tconsts:
        if dm:
            # Check which liked_tconsts are not in all_titles_df
            existing_liked = set(all_titles_df.filter(pl.col("tconst").is_in(liked_tconsts))["tconst"].to_list())
            missing_liked = [t for t in liked_tconsts if t not in existing_liked]
            if missing_liked:
                parent_map = dm.resolve_parents(missing_liked)
                # Map episode tconsts to their parent series tconsts
                liked_tconsts = [parent_map.get(t, t) for t in liked_tconsts]

        liked_df = all_titles_df.filter(pl.col("tconst").is_in(liked_tconsts))
        
        for genres_str in liked_df["genres"].to_list():
            if genres_str:
                for g in genres_str.split(","):
                    if g in all_genres:
                        genre_counts[all_genres.index(g)] = 1.0 # Give stronger weight to implicitly liked genres
        
        avg_runtime = liked_df["runtimeMinutes"].fill_null(0).mean() / 300.0
        avg_year = (liked_df["startYear"].fill_null(2000).mean() - 1900) / 150.0
    else:
        avg_runtime = 0.0
        avg_year = 0.0
    
    # Normalize genre vector
    if np.sum(genre_counts) > 0:
        genre_counts /= np.sum(genre_counts)
        
    user_vec = np.hstack([genre_counts, [avg_runtime, avg_year]])
    return user_vec.astype('float32').reshape(1, -1)

