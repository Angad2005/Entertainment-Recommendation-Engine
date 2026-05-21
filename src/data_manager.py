import polars as pl
import sqlite3
import requests
import gzip
import shutil
import os
from pathlib import Path
from datetime import datetime, timedelta

class DataManager:
    IMDB_URLS = {
        "basics": "https://datasets.imdbws.com/title.basics.tsv.gz",
        "ratings": "https://datasets.imdbws.com/title.ratings.tsv.gz",
        "episodes": "https://datasets.imdbws.com/title.episode.tsv.gz"
    }

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "profiles.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profiles (
                    name TEXT PRIMARY KEY,
                    genres TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Re-create table for new schema (separating Preference and Watched status)
            conn.execute("DROP TABLE IF EXISTS reviews")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    profile_name TEXT,
                    tconst TEXT,
                    preference TEXT, -- 'like', 'dislike'
                    watched INTEGER DEFAULT 0, -- 1 for watched, 0 for not
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(profile_name, tconst),
                    FOREIGN KEY(profile_name) REFERENCES profiles(name)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

    def download_data(self, force=False, callback=None):
        for name, url in self.IMDB_URLS.items():
            dest = self.data_dir / f"{name}.tsv.gz"
            if not dest.exists() or force:
                print(f"Downloading {name}...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(dest, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192 * 1024): # 8MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if callback and total_size > 0:
                                callback(name, downloaded / total_size)
        
        # Update last refresh time
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", 
                         ("last_refresh", datetime.now().isoformat()))

    def check_refresh_needed(self):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT value FROM meta WHERE key = 'last_refresh'").fetchone()
            if not res:
                return True
            last_refresh = datetime.fromisoformat(res[0])
            return datetime.now() - last_refresh > timedelta(days=7)

    def load_data(self):
        basics_path = self.data_dir / "basics.tsv.gz"
        ratings_path = self.data_dir / "ratings.tsv.gz"

        if not basics_path.exists() or not ratings_path.exists():
            self.download_data()

        # Use Lazy Loading with scan_csv
        basics_lazy = pl.scan_csv(basics_path, separator="\t", null_values="\\N", ignore_errors=True, quote_char=None)
        ratings_lazy = pl.scan_csv(ratings_path, separator="\t", null_values="\\N", ignore_errors=True, quote_char=None)

        # Apply filters and joins lazily
        df_lazy = basics_lazy.filter(pl.col("titleType").is_in(["movie", "tvSeries"]))
        df_lazy = df_lazy.join(ratings_lazy, on="tconst", how="inner")
        
        # Select only necessary columns to save memory
        cols = ["tconst", "titleType", "primaryTitle", "startYear", "runtimeMinutes", "genres", "averageRating", "numVotes"]
        df_lazy = df_lazy.select(cols)
        
        # Collect into memory
        return df_lazy.collect()

    def create_profile(self, name, genres):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("INSERT INTO profiles (name, genres) VALUES (?, ?)", 
                             (name, ",".join(genres)))
            return True
        except sqlite3.IntegrityError:
            return False

    def get_profile(self, name):
        with sqlite3.connect(self.db_path) as conn:
            res = conn.execute("SELECT * FROM profiles WHERE name = ?", (name,)).fetchone()
            if res:
                return {"name": res[0], "genres": res[1].split(",")}
            return None

    def update_preference(self, profile_name, tconst, preference):
        with sqlite3.connect(self.db_path) as conn:
            # Check if entry exists
            cursor = conn.execute("SELECT 1 FROM reviews WHERE profile_name = ? AND tconst = ?", (profile_name, tconst))
            if cursor.fetchone():
                conn.execute("UPDATE reviews SET preference = ? WHERE profile_name = ? AND tconst = ?",
                             (preference, profile_name, tconst))
            else:
                conn.execute("INSERT INTO reviews (profile_name, tconst, preference) VALUES (?, ?, ?)",
                             (profile_name, tconst, preference))

    def toggle_watched(self, profile_name, tconst):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT watched FROM reviews WHERE profile_name = ? AND tconst = ?", (profile_name, tconst))
            res = cursor.fetchone()
            if res:
                new_state = 0 if res[0] == 1 else 1
                conn.execute("UPDATE reviews SET watched = ? WHERE profile_name = ? AND tconst = ?",
                             (new_state, profile_name, tconst))
            else:
                conn.execute("INSERT INTO reviews (profile_name, tconst, watched) VALUES (?, ?, 1)",
                             (profile_name, tconst))

    def get_user_history(self, profile_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT tconst, preference, watched FROM reviews WHERE profile_name = ?", (profile_name,))
            return {row[0]: {"preference": row[1], "watched": row[2]} for row in cursor.fetchall()}

    def delete_profile(self, name):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM reviews WHERE profile_name = ?", (name,))
            conn.execute("DELETE FROM profiles WHERE name = ?", (name,))

    def get_episodes(self, parent_tconst):
        ep_path = self.data_dir / "episodes.tsv.gz"
        basics_path = self.data_dir / "basics.tsv.gz"
        
        # Scan episodes and filter by parent
        eps_lazy = pl.scan_csv(ep_path, separator="\t", null_values="\\N", ignore_errors=True)
        relevant_eps = eps_lazy.filter(pl.col("parentTconst") == parent_tconst)
        
        # Join with basics to get titles
        basics_lazy = pl.scan_csv(basics_path, separator="\t", null_values="\\N", ignore_errors=True, quote_char=None)
        
        full_eps = relevant_eps.join(basics_lazy, left_on="tconst", right_on="tconst", how="inner")
        full_eps = full_eps.select(["tconst", "parentTconst", "seasonNumber", "episodeNumber", "primaryTitle"])
        
        df = full_eps.collect()
        df = df.with_columns([
            pl.col("seasonNumber").cast(pl.Int32, strict=False).fill_null(0),
            pl.col("episodeNumber").cast(pl.Int32, strict=False).fill_null(0)
        ])
        return df.sort(["seasonNumber", "episodeNumber"]).to_dicts()

    def get_popular_titles_by_genres(self, genres, limit=5):
        df = self.load_data()
        # Filter by genres and sort by rating count/average
        filtered = df.filter(pl.col("genres").str.contains("|".join(genres)))
        popular = filtered.sort("numVotes", descending=True).head(limit)
        return popular.to_dicts()
