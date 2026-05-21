import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from pathlib import Path
from data_manager import DataManager
from model import TwoTowerModel, get_device, save_model, load_model
from recommender import Recommender, prepare_item_features, prepare_user_features
from trainer import AsyncTrainer

# --- Page Config ---
st.set_page_config(page_title="AI Movie & TV Engine", page_icon="🎬", layout="wide")

# --- UI Aesthetics ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { border-radius: 20px; transition: 0.3s; }
    .stButton>button:hover { transform: scale(1.05); background-color: #ff4b4b; color: white; }
    .card { background-color: #1e2130; padding: 20px; border-radius: 15px; margin-bottom: 10px; border: 1px solid #3e4150; }
    .title-text { color: #ff4b4b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Initialization ---
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
    st.session_state.all_genres = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
        "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Short", "Sport", "Thriller", "War", "Western"
    ]
    st.session_state.device = get_device()

dm = st.session_state.data_manager

# --- Sidebar: Profile & GPU Status ---
with st.sidebar:
    st.title("🎬 Engine Status")
    gpu_available = torch.cuda.is_available()
    st.status(f"CUDA: {'✅ Enabled' if gpu_available else '❌ CPU Only'}", expanded=False)
    
    if st.session_state.get('profile'):
        st.write(f"Logged in as: **{st.session_state.profile['name']}**")
        if st.button("🗑️ Delete Profile"):
            dm.delete_profile(st.session_state.profile['name'])
            st.session_state.profile = None
            st.rerun()
        if st.button("🚪 Logout"):
            st.session_state.profile = None
            st.rerun()

# --- App Flow ---

def main():
    check_data_and_load()
    if 'profile' not in st.session_state or st.session_state.profile is None:
        show_login()
    else:
        show_dashboard()

def show_login():
    st.title("Welcome to the Hybrid Movie Engine")
    st.markdown("### Identify yourself to begin personalizing your experience.")
    
    name = st.text_input("Enter Profile Name", key="login_name")
    if st.button("Start Engine"):
        if name:
            profile = dm.get_profile(name)
            if profile:
                st.session_state.profile = profile
                st.rerun()
            else:
                st.session_state.new_profile_name = name
                st.session_state.onboarding_step = 1
                st.rerun()
    
    if 'onboarding_step' in st.session_state:
        show_onboarding()

def show_onboarding():
    st.divider()
    if st.session_state.onboarding_step == 1:
        st.subheader("Step 1: Choose your favorite genres")
        selected = st.multiselect("Select at least 3", st.session_state.all_genres)
        if st.button("Next") and len(selected) >= 3:
            st.session_state.selected_genres = selected
            st.session_state.onboarding_step = 2
            st.rerun()
            
    elif st.session_state.onboarding_step == 2:
        st.subheader("Step 2: Personalize your Profile")
        st.write("Tell us what you've seen and what you like. This builds your initial embedding.")
        if 'onboarding_titles' not in st.session_state:
            with st.spinner("Curating titles for you..."):
                df = get_cached_data()
                genres_regex = "|".join(st.session_state.selected_genres)
                filtered = df.filter(pl.col("genres").str.contains(genres_regex))
                # Get more titles for a robust profile
                popular = filtered.sort("numVotes", descending=True).head(10)
                st.session_state.onboarding_titles = popular.to_dicts()
                st.session_state.onboarding_feedback = {}

        for title in st.session_state.onboarding_titles:
            tconst = title['tconst']
            col1, col2 = st.columns([3, 2])
            col1.write(f"**{title['primaryTitle']}** ({title['startYear']})")
            
            # Button group for feedback
            fb_cols = col2.columns(3)
            if fb_cols[0].button("👍 Like", key=f"like_{tconst}"):
                st.session_state.onboarding_feedback[tconst] = st.session_state.onboarding_feedback.get(tconst, {"preference": None, "watched": 0})
                st.session_state.onboarding_feedback[tconst]["preference"] = "like"
            if fb_cols[1].button("👎 Dislike", key=f"dis_{tconst}"):
                st.session_state.onboarding_feedback[tconst] = st.session_state.onboarding_feedback.get(tconst, {"preference": None, "watched": 0})
                st.session_state.onboarding_feedback[tconst]["preference"] = "dislike"
            if fb_cols[2].button("👀 Seen", key=f"seen_{tconst}"):
                st.session_state.onboarding_feedback[tconst] = st.session_state.onboarding_feedback.get(tconst, {"preference": None, "watched": 0})
                st.session_state.onboarding_feedback[tconst]["watched"] = 1
            
            # Show current selection
            if tconst in st.session_state.onboarding_feedback:
                pref = st.session_state.onboarding_feedback[tconst].get('preference')
                wat = st.session_state.onboarding_feedback[tconst].get('watched')
                status = f"{pref if pref else ''} {'(Seen)' if wat else ''}"
                col2.caption(status)

        # Show feedback progress
        liked_count = sum(1 for v in st.session_state.onboarding_feedback.values() if v.get('preference') == 'like')
        disliked_count = sum(1 for v in st.session_state.onboarding_feedback.values() if v.get('preference') == 'dislike')
        watched_count = sum(1 for v in st.session_state.onboarding_feedback.values() if v.get('watched') == 1)
        
        st.info(f"Progress: 👍 {liked_count} | 👎 {disliked_count} | 👀 {watched_count} (Minimum 3 total recommended)")

        if st.button("🚀 Finalize Profile"):
            total_feedback = liked_count + disliked_count + watched_count
            if total_feedback < 3:
                st.warning("Please provide feedback for at least 3 titles.")
            else:
                dm.create_profile(st.session_state.new_profile_name, st.session_state.selected_genres)
                for tconst, fb in st.session_state.onboarding_feedback.items():
                    if fb.get('preference'):
                        dm.update_preference(st.session_state.new_profile_name, tconst, fb['preference'])
                    if fb.get('watched'):
                        dm.toggle_watched(st.session_state.new_profile_name, tconst)
                
                st.session_state.profile = dm.get_profile(st.session_state.new_profile_name)
                st.success("Profile initialized! Entering Engine...")
                time.sleep(1)
                st.rerun()

@st.fragment
def training_ui(trainer):
    if trainer.is_training:
        st.write(f"**Status:** {trainer.status}")
        st.progress(trainer.progress)
        if st.button("🚫 Cancel Update"):
            trainer.cancel()
    elif trainer.status == "Training complete.":
        st.success("Model updated successfully!")
    elif "cancelled" in trainer.status:
        st.warning(trainer.status)

def check_data_and_load():
    basics_path = Path("data/basics.tsv.gz")
    ratings_path = Path("data/ratings.tsv.gz")
    episodes_path = Path("data/episodes.tsv.gz")
    
    if not basics_path.exists() or not ratings_path.exists() or not episodes_path.exists():
        st.info("🚀 First run detected. Downloading IMDb dataset (this may take a few minutes)...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(name, progress):
            status_text.text(f"Downloading {name}: {int(progress * 100)}%")
            progress_bar.progress(progress)
            
        dm.download_data(callback=update_progress)
        st.success("Download complete!")
        time.sleep(1)
        st.rerun()

@st.cache_data(show_spinner=False)
def get_cached_data():
    return dm.load_data()

def show_dashboard():
    # Load Data & Model
    if 'df' not in st.session_state:
        with st.spinner("Loading IMDb Data (Polars Accelerated)..."):
            st.session_state.df = get_cached_data()
            st.session_state.item_features = prepare_item_features(st.session_state.df, st.session_state.all_genres)
            
    if 'model' not in st.session_state:
        input_dim = len(st.session_state.all_genres) + 2
        st.session_state.model = TwoTowerModel(input_dim, input_dim).to(st.session_state.device)
        st.session_state.optimizer = optim.Adam(st.session_state.model.parameters(), lr=0.001)
        st.session_state.criterion = nn.MSELoss()
        st.session_state.trainer = AsyncTrainer(st.session_state.model, st.session_state.optimizer, st.session_state.criterion)
        st.session_state.recommender = Recommender(st.session_state.model, st.session_state.df)
        st.session_state.recommender.build_index(st.session_state.item_features)

    st.title(f"Hi, {st.session_state.profile['name']}! 🎥")
    
    # Weekly check
    if dm.check_refresh_needed():
        st.warning("New IMDb data is available! A refresh is recommended.")
        if st.button("Refresh Now"):
            dm.download_data(force=True)
            st.rerun()

    # Async Training Monitor
    training_ui(st.session_state.trainer)

    # Recommendations
    st.subheader("Recommended for You")
    user_history = dm.get_user_history(st.session_state.profile['name'])
    user_feat = prepare_user_features(
        user_history, 
        st.session_state.df, 
        st.session_state.all_genres,
        st.session_state.profile.get('genres', [])
    )
    
    recs = st.session_state.recommender.recommend(
        user_feat, 
        exclude_tconsts=[t for t, v in user_history.items() if v.get('watched') == 1]
    )
    
    cols = st.columns(4)
    for i, rec in enumerate(recs[:8]):
        tconst = rec['tconst']
        with cols[i % 4]:
            st.markdown(f"""
                <div class="card">
                    <h4 class="title-text">{rec['primaryTitle']}</h4>
                    <p>IMDb: ⭐ {rec['averageRating']}</p>
                    <p>Match: {int(rec['score']*100)}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            # Highlighting active status
            is_liked = user_history.get(tconst, {}).get('preference') == 'like'
            is_disliked = user_history.get(tconst, {}).get('preference') == 'dislike'
            is_watched = user_history.get(tconst, {}).get('watched') == 1

            if c1.button("👍" if not is_liked else "🌟", key=f"l_{tconst}"):
                dm.update_preference(st.session_state.profile['name'], tconst, "like")
                item_idx = np.where(st.session_state.df["tconst"].to_numpy() == tconst)[0][0]
                st.session_state.trainer.start_training(user_feat, st.session_state.item_features[item_idx].reshape(1,-1), ["like"])
                st.rerun()
            if c2.button("👎" if not is_disliked else "🚫", key=f"d_{tconst}"):
                dm.update_preference(st.session_state.profile['name'], tconst, "dislike")
                item_idx = np.where(st.session_state.df["tconst"].to_numpy() == tconst)[0][0]
                st.session_state.trainer.start_training(user_feat, st.session_state.item_features[item_idx].reshape(1,-1), ["dislike"])
                st.rerun()
            if c3.button("👀" if not is_watched else "✅", key=f"v_{tconst}"):
                dm.toggle_watched(st.session_state.profile['name'], tconst)
                st.rerun()
            
            # Show Episode Breakdown for TV Series
            # We check if it's a tvSeries in the main dataframe
            item_row = st.session_state.df.filter(pl.col("tconst") == rec['tconst']).to_dicts()[0]
            if item_row['titleType'] == 'tvSeries':
                with st.expander("📺 Episodes"):
                    episodes = dm.get_episodes(rec['tconst'])
                    if episodes:
                        current_season = None
                        for ep in episodes:
                            if ep['seasonNumber'] != current_season:
                                current_season = ep['seasonNumber']
                                st.markdown(f"**Season {current_season}**")
                            
                            ec1, ec2 = st.columns([3, 2])
                            ec1.caption(f"E{ep['episodeNumber']}: {ep['primaryTitle']}")
                            
                            is_ep_liked = user_history.get(ep['tconst'], {}).get('preference') == 'like'
                            is_ep_disliked = user_history.get(ep['tconst'], {}).get('preference') == 'dislike'
                            is_ep_watched = user_history.get(ep['tconst'], {}).get('watched') == 1

                            efb = ec2.columns(3)
                            if efb[0].button("👍" if not is_ep_liked else "🌟", key=f"ep_l_{ep['tconst']}"):
                                dm.update_preference(st.session_state.profile['name'], ep['tconst'], "like")
                                st.rerun()
                            if efb[1].button("👎" if not is_ep_disliked else "🚫", key=f"ep_d_{ep['tconst']}"):
                                dm.update_preference(st.session_state.profile['name'], ep['tconst'], "dislike")
                                st.rerun()
                            if efb[2].button("👀" if not is_ep_watched else "✅", key=f"ep_s_{ep['tconst']}"):
                                dm.toggle_watched(st.session_state.profile['name'], ep['tconst'])
                                st.rerun()
                    else:
                        st.write("No episode data available.")

    # Search & Explore
    st.divider()
    st.subheader("Explore All Titles")
    search_q = st.text_input("Search Movie/TV Title")
    if search_q:
        results = st.session_state.df.filter(pl.col("primaryTitle").str.to_lowercase().str.contains(search_q.lower())).head(10)
        for row in results.to_dicts():
            c1, c2 = st.columns([3, 2])
            c1.write(f"**{row['primaryTitle']}** ({row['startYear']})")
            
            # Highlight active status in search results
            is_liked = user_history.get(row['tconst'], {}).get('preference') == 'like'
            is_disliked = user_history.get(row['tconst'], {}).get('preference') == 'dislike'
            is_watched = user_history.get(row['tconst'], {}).get('watched') == 1

            fb_cols = c2.columns(3)
            if fb_cols[0].button("👍" if not is_liked else "🌟", key=f"s_like_{row['tconst']}"):
                dm.update_preference(st.session_state.profile['name'], row['tconst'], "like")
                item_idx = np.where(st.session_state.df["tconst"].to_numpy() == row['tconst'])[0][0]
                st.session_state.trainer.start_training(user_feat, st.session_state.item_features[item_idx].reshape(1,-1), ["like"])
                st.rerun()
            if fb_cols[1].button("👎" if not is_disliked else "🚫", key=f"s_dis_{row['tconst']}"):
                dm.update_preference(st.session_state.profile['name'], row['tconst'], "dislike")
                item_idx = np.where(st.session_state.df["tconst"].to_numpy() == row['tconst'])[0][0]
                st.session_state.trainer.start_training(user_feat, st.session_state.item_features[item_idx].reshape(1,-1), ["dislike"])
                st.rerun()
            if fb_cols[2].button("👀" if not is_watched else "✅", key=f"s_seen_{row['tconst']}"):
                dm.toggle_watched(st.session_state.profile['name'], row['tconst'])
                st.rerun()
                
            # Show Episode Breakdown for TV Series in search results
            if row['titleType'] == 'tvSeries':
                with st.expander("📺 Episodes"):
                    episodes = dm.get_episodes(row['tconst'])
                    if episodes:
                        current_season = None
                        for ep in episodes:
                            if ep['seasonNumber'] != current_season:
                                current_season = ep['seasonNumber']
                                st.markdown(f"**Season {current_season}**")
                            
                            ec1, ec2 = st.columns([3, 2])
                            ec1.caption(f"E{ep['episodeNumber']}: {ep['primaryTitle']}")
                            
                            is_ep_liked = user_history.get(ep['tconst'], {}).get('preference') == 'like'
                            is_ep_disliked = user_history.get(ep['tconst'], {}).get('preference') == 'dislike'
                            is_ep_watched = user_history.get(ep['tconst'], {}).get('watched') == 1

                            efb = ec2.columns(3)
                            if efb[0].button("👍" if not is_ep_liked else "🌟", key=f"sep_l_{ep['tconst']}"):
                                dm.update_preference(st.session_state.profile['name'], ep['tconst'], "like")
                                st.rerun()
                            if efb[1].button("👎" if not is_ep_disliked else "🚫", key=f"sep_d_{ep['tconst']}"):
                                dm.update_preference(st.session_state.profile['name'], ep['tconst'], "dislike")
                                st.rerun()
                            if efb[2].button("👀" if not is_ep_watched else "✅", key=f"sep_s_{ep['tconst']}"):
                                dm.toggle_watched(st.session_state.profile['name'], ep['tconst'])
                                st.rerun()
                    else:
                        st.write("No episode data available.")

if __name__ == "__main__":
    main()
