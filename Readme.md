# 🎬 AI Hybrid Movie & TV Recommendation Engine (Human‑Friendly Overview)

## What is this project?
It’s a **movie‑recommendation app** that suggests films and TV shows you might like.  It uses a small neural network (called a *Two‑Tower model*) to turn items (movies/episodes) and users into numbers, then finds the items whose numbers are closest to yours.

## Why use it?
- **Fast** – When a compatible NVIDIA GPU is present, we use *FAISS* to search billions of vectors in milliseconds.
- **Accurate** – The model combines what you’ve watched/liked (collaborative) with the movie’s metadata (genre, year, rating) for a hybrid score.
- **Simple to run** – One command launches a Streamlit web UI that lets you create a profile, give feedback, and see recommendations.

## Key ideas (no jargon)
| Concept | Plain language |
|---------|----------------|
| Two‑Tower model | Two separate calculators: one turns a **user’s history** into a vector, the other turns a **movie’s details** into a vector. The vectors are then compared.
| FAISS index | A clever data structure that can find the *nearest* vectors super quickly (think “find the closest friends”).
| Polars | A fast table library (like pandas) that lets us read the huge IMDb files without using too much memory.
| Hybrid scoring | We mix similarity (how close the vectors are) with the movie’s average rating, so a well‑rated film can move up the list.

## How the code is organized
```
Engine/
├─ data/                 # IMDb raw TSV.gz files and a SQLite DB for profiles
├─ src/                  # Core Python code
│   ├─ app.py            # Streamlit UI – the dashboard you’ll see in the browser
│   ├─ data_manager.py   # Loads IMDb data, handles the SQLite profile store
│   ├─ model.py          # Definition of the Two‑Tower PyTorch model
│   ├─ recommender.py    # Builds the FAISS index and performs the hybrid search
│   └─ trainer.py        # Background worker that updates the model when you give new feedback
├─ requirements.txt      # Packages you need to install
└─ Readme.md             # This file (human‑friendly version)
```

## Quick start (step‑by‑step)
1. **Clone the repo**
   ```bash
   git clone <repo‑url>
   cd Engine
   ```
2. **Install the Python packages**
   ```bash
   pip install -r requirements.txt
   ```
3. **Check your hardware** – If you have an NVIDIA GPU with CUDA installed, the engine will automatically use it. If not, it will fall back to the CPU.
4. **Run the web app**
   ```bash
   streamlit run src/app.py
   ```
   Open the URL shown in the terminal (usually `http://localhost:8501`).

## First‑time experience
- **Create a profile** – Choose a name. The app will ask you to pick a few favorite genres.
- **On‑boarding** – You’ll see a short list of titles and can mark them as *Liked*, *Disliked*, or *Seen*. This tiny amount of feedback is enough for the model to build an initial user vector.
- **Explore** – Use the search bar to browse titles, click the "👍 Like" or "👀 Seen" buttons, and watch the recommendations update in real time.

## Updating the model
When you give new likes/dislikes, a background thread (`trainer.AsyncTrainer`) retrains the Two‑Tower model without freezing the UI. Once training finishes, the index is rebuilt automatically.

## FAQs
**Do I need a GPU?** No. The code works on CPU, just slower. It will automatically detect CUDA.

**Can I add my own movies?** The data loader works with the official IMDb TSV files. You could extend `data_manager.py` to load a custom CSV if you wish.

**What if I don’t have Streamlit installed?** It’s listed in `requirements.txt`. Installing the requirements will bring it in.

## License
MIT – feel free to adapt, share, and build on this project.

---
*Enjoy discovering your next favorite show!* 🍿
