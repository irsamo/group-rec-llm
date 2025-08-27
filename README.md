# Hybrid Group Book Recommender

## 🚀 Overview

This project explores and evaluates different strategies for recommending books to groups of users. The core motivation is to demonstrate that a **hybrid recommendation model**, which combines the mathematical rigor of traditional Collaborative Filtering (CF) with the contextual understanding of a Large Language Model (LLM), can outperform either method in isolation.

The application provides a head-to-head comparison of three models:
1.  **Pure Collaborative Filtering**: A classic `surprise SVD` model that uses user rating data to find recommendations.
2.  **Pure LLM**: A `GPT-4` based model that uses natural language descriptions of user preferences to make recommendations.
3.  **Hybrid (CF + LLM)**: A sophisticated model that uses the CF model to generate a list of promising candidates and then uses the LLM to re-rank those candidates to find the best fit for the group.

An interactive Streamlit application runs a quantitative evaluation to test these models against a "ground truth" and prove the effectiveness of the hybrid approach.

## ✨ Features

- **Three Recommendation Models**: Implements Pure CF, Pure LLM, and a Hybrid model.
- **Group Recommendation Strategies**: Allows for different group satisfaction goals, including:
    - `average`: Aims for overall group satisfaction.
    - `least_misery`: Tries to avoid books any single member might dislike.
    - `happy_scenario`: Focuses on books that at least one member will absolutely love.
- **Quantitative Evaluation**: A Streamlit app runs a "blind test" to see if the models can predict a book that a group is known to like, providing a clear success/failure metric.
- **Model Persistence**: The trained CF model is saved to disk (`svd_model.pkl`) to avoid retraining on every run.
- **Data**: Uses the well-known [Book-Crossing dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

## ✅ Requirements

- **Conda** (Anaconda / Miniforge / Mambaforge)
- Internet access for Python packages and the OpenAI API
- CSVs from the **Book-Crossing** dataset

> Python and packages are pinned in `environment.yml` (Python 3.11).

---

## ⚙️ Setup

### 1) Create/update the Conda environment

```bash
# first time
conda env create -f environment.yml
conda activate grouprec

# later, to sync with environment.yml (removes extras)
conda activate grouprec
conda env update --file environment.yml --prune
```

> Tip: keep your `pip:` section at the end of `environment.yml`. Packages listed there (e.g., `openai`, `python-dotenv`) are installed automatically when you create/update the env.

---

### 2) Configure the OpenAI API key (no keys in code)

The OpenAI Python client reads your key from the environment variable ``.\
**Never** paste real keys into code or docs.


1. Create a file named `.env` in the project root:

```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

2. Do **not** commit `.env`. (It is ignored by `.gitignore`.)

3. The app loads it automatically via `python-dotenv` (already included in `environment.yml`).

> If you want to switch models, you can add another env var, e.g. `OPENAI_MODEL=gpt-4o`, and read it in your code (optional).


## ▶️ Run

### Train the CF model once

```bash
python collaborative_filtering_bookcrossing.py
```

This builds and caches the SVD model so you don’t retrain on every run.

### Launch the Streamlit app

```bash
streamlit run app.py
# or:
# python -m streamlit run app.py
```
 *   Open the provided URL in your browser, select a group recommendation strategy from the sidebar, and click "Run Evaluation Demo".

## 🧩 Repository contents (key files)

- `app.py` — Streamlit UI for comparing strategies and models.
- `collaborative_filtering_bookcrossing.py` — trains / saves the SVD CF model.
- `openai_group_recommender.py` — LLM wrapper (reads `OPENAI_API_KEY` from env).
- `comparison_recommender.py` — evaluation/comparison routines.
- `memory_store.py`, `seed_memory_from_dataset.py` — optional memory utilities.
- `user_prefs_kv.json`, `user_prefs_meta.json`, `user_prefs.faiss` — cached preferences/embeddings (if used).
- `environment.yml` — reproducible Conda environment (includes `python-dotenv`).
- `.env` — **not committed**; put `OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>` here for local runs.
- `.gitignore` — ignores `.env`, `__pycache__/`, `.DS_Store`, IDE folders, etc.

---

*(Note: The `BX-Book-Ratings.csv` and `BX-Books.csv` data files are required to run the project but are not included in this repository).*
