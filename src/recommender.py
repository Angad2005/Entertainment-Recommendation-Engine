import faiss
import numpy as np
import torch
import polars as pl
from model import get_device

class Recommender:
    def __init__(self, model, item_df, embedding_dim=128):
        self.model = model
        self.item_df = item_df
        self.embedding_dim = embedding_dim
        self.device = get_device()
        self.index = None
        self.item_embeddings = None

    def build_index(self, item_features):
        """
        Build FAISS index. Uses GpuIndexFlatL2 if GPU is detected.
        """
        item_features_tensor = torch.FloatTensor(item_features).to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.item_embeddings = self.model.get_item_embedding(item_features_tensor).cpu().numpy()

        d = self.embedding_dim
        res = faiss.StandardGpuResources() if torch.cuda.is_available() else None
        
        if res:
            # GPU Index optimized for speed
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0
            flat_config.useFloat16 = True # Maximum speed on RTX GPUs
            self.index = faiss.GpuIndexFlatL2(res, d, flat_config)
        else:
            # CPU Index
            self.index = faiss.IndexFlatL2(d)

        self.index.add(self.item_embeddings.astype('float32'))

    def recommend(self, user_features, exclude_tconsts=None, alpha=0.7, beta=0.3, top_k=20):
        """
        Hybrid recommendation logic with exclusion of already watched titles.
        """
        user_features_tensor = torch.FloatTensor(user_features).to(self.device)
        self.model.eval()
        with torch.no_grad():
            user_embedding = self.model.get_user_embedding(user_features_tensor).cpu().numpy()

        # FAISS search
        distances, indices = self.index.search(user_embedding.astype('float32'), top_k * 5) # Get more to allow filtering
        
        results = []
        exclude_set = set(exclude_tconsts) if exclude_tconsts else set()
        
        for dist, idx in zip(distances[0], indices[0]):
            item = self.item_df.row(idx, named=True)
            if item["tconst"] in exclude_set:
                continue
                
            # Sim score (inverse of distance for L2)
            sim_score = 1 / (1 + dist)
            avg_rating = item.get("averageRating", 0) / 10.0 # Normalize 1-10 to 0-1
            
            hybrid_score = alpha * sim_score + beta * avg_rating
            results.append({
                "tconst": item["tconst"],
                "primaryTitle": item["primaryTitle"],
                "score": hybrid_score,
                "averageRating": item["averageRating"]
            })
            if len(results) >= top_k:
                break
            
        # Sort by hybrid score
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

def prepare_user_features(user_history, all_titles_df, all_genres, user_selected_genres=None):
    """
    Build user feature vector from history and explicitly selected genres.
    """
    user_selected_genres = user_selected_genres or []
    genre_counts = np.zeros(len(all_genres))
    
    # Base weights from explicitly selected genres
    for g in user_selected_genres:
        if g in all_genres:
            genre_counts[all_genres.index(g)] += 0.5 # Give base weight to selected genres

    if not user_history:
        # Default/Cold start features
        if np.sum(genre_counts) > 0:
            genre_counts /= np.sum(genre_counts)
        features = np.hstack([genre_counts, np.array([0.0, 0.0])])
        return features.astype('float32').reshape(1, -1)
    
    # Aggregate genres of liked items
    liked_tconsts = [t for t, v in user_history.items() if v.get('preference') == 'like']
    
    if liked_tconsts:
        liked_df = all_titles_df.filter(pl.col("tconst").is_in(liked_tconsts))
        
        for genres_str in liked_df["genres"].to_list():
            if genres_str:
                for g in genres_str.split(","):
                    if g in all_genres:
                        genre_counts[all_genres.index(g)] += 1.0 # Give stronger weight to implicitly liked genres
        
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
