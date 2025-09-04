import os
import math
import re
import random
from collections import defaultdict
from statistics import fmean

import pandas as pd
import pickle
import streamlit as st
from openai import OpenAI

# local
from memory_store import get_prefs_exact, retrieve_similar_prefs

# ===== Oracle evaluation settings =====
LAMBDA_SHRINK = 20        # prior strength for shrinkage toward global mean (IMDb-style)  
RNG_SEED = 42             # for reproducible splits

# ===== LLM safety caps =====
LLM_MODEL = "gpt-4"          # or a newer model with a larger window, e.g., "gpt-4o"
LLM_MAX_TOKENS = 512         # keep outputs short so more room for the prompt
MAX_TITLES_FOR_LLM = 150     # cap how many candidate titles you show the LLM
MAX_LIKES_PER_USER = 5       # cap how many "likes" per user you include


# --- App Configuration ---
st.set_page_config(page_title="Recommender System Evaluation", layout="wide")


# --- Caching Functions for Efficiency ---

@st.cache_resource
def load_cf_model(model_path='svd_model.pkl'):
    if not os.path.exists(model_path):
        st.error(f"Error: Saved model '{model_path}' not found. Please run 'collaborative_filtering_bookcrossing.py' first.")
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


@st.cache_data
def load_data():
    try:
        ratings_df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        books_df = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        return ratings_df, books_df
    except FileNotFoundError as e:
        st.error(f"Error: Could not find a required data file: {e}")
        return None, None
    

# --- Functions for parsing and scoring ---
# --- Normalization (single source of truth) ---
_punct_re = re.compile(r"[^\w]+", flags=re.UNICODE)

def _norm_title(s: str) -> str:
    """Lowercase + strip punctuation + collapse spaces (robust title equality)."""
    if s is None:
        return ""
    s = _punct_re.sub(" ", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# Backwards-compat alias (if other parts still call _normalize_title)
_normalize_title = _norm_title

# --- LLM list parser (kept; dedupe via robust normalizer) ---
def extract_titles_from_llm_output(text: str):
    """
    Parse LLM's numbered list into a Python list of titles.
    Supports '1. Title', '2) Title', '3 - Title'. Falls back to comma/newline split.
    Dedupe while preserving order.
    """
    titles = []
    for line in str(text).splitlines():
        m = re.match(r'^\s*\d+[\).\s-]+(.+)$', line.strip())
        if m:
            titles.append(m.group(1).strip())
    if not titles:
        # fallback: comma/newline separated
        parts = re.split(r',|\n', str(text))
        titles = [p.strip(" -") for p in parts if p.strip()]

    # dedupe using robust normalizer
    seen, out = set(), []
    for t in titles:
        k = _norm_title(t)
        if k and k not in seen:
            seen.add(k)
            out.append(t)
    return out

# ===== Small helpers for graded DCG/nDCG =====
def dcg_at_k(gains: list[float]) -> float:
    """Graded DCG using log2 discount."""
    s = 0.0
    for idx, g in enumerate(gains, start=1):
        if g <= 0: 
            continue
        s += g / math.log2(idx + 1)
    return s

def ndcg_at_k_graded(pred_items: list[str], gain_by_item: dict, k: int) -> float:
    """
    nDCG@K with graded gains:
      - pred_items: list of *items* (we'll use ISBNs as keys)
      - gain_by_item: dict ISBN -> gain in [0,1]
    """
    top = pred_items[:k]
    gains = [float(gain_by_item.get(it, 0.0)) for it in top]
    dcg  = dcg_at_k(gains)

    # Ideal gains (sorted) for the same K
    ideal_gains = sorted(gain_by_item.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_gains)
    return (dcg / idcg) if idcg > 0 else 0.0

def ndcg_at_k_binary(pred_items: list[str], oracle_topk: list[str], k: int) -> float:
    """
    Binary-relevance nDCG@K against the Oracle Top-K set.
    Gain = 1 if item ∈ OracleTopK, else 0. Uses the same DCG discount as before.
    """
    top_pred = list(map(str, pred_items[:k]))
    rel_set = set(map(str, oracle_topk[:k]))
    gains = [1.0 if it in rel_set else 0.0 for it in top_pred]
    dcg  = dcg_at_k(gains)
    idcg = dcg_at_k([1.0] * min(k, len(rel_set)))
    return (dcg / idcg) if idcg > 0 else 0.0


# ----------------------------------------------

# ===== Train/Test split for the GROUP (ratings, not titles) =====
def split_ratings_train_test_for_group(ratings_df: pd.DataFrame, group_users: list[int], test_frac: float = 0.2, seed: int = RNG_SEED):
    """
    Random 80/20 split *per group user* over their ratings rows.
    Returns TRAIN_DF, TEST_DF (disjoint row sets).
    """
    rng = random.Random(seed)
    all_idx = []
    for u in group_users:
        user_idxs = ratings_df.index[ratings_df['User-ID'] == u].tolist()
        rng.shuffle(user_idxs)
        n_test = max(1, int(len(user_idxs) * test_frac))
        all_idx.extend(user_idxs[:n_test])
    test_df  = ratings_df.loc[sorted(set(all_idx))]
    train_df = ratings_df.drop(index=test_df.index)
    return train_df, test_df

# ===== Item statistics from TRAIN (for shrinkage) =====
def compute_item_stats(train_df: pd.DataFrame):
    """
    Returns:
      item_mean: dict ISBN -> mean rating in TRAIN
      item_count: dict ISBN -> count in TRAIN
      mu: global mean in TRAIN
    """
    grp = train_df.groupby('ISBN')['Book-Rating']
    item_mean  = grp.mean().to_dict()
    item_count = grp.count().to_dict()
    mu = float(train_df['Book-Rating'].mean())
    return item_mean, item_count, mu

def impute_item_shrink(isbn, item_mean, item_count, mu, lam=LAMBDA_SHRINK, n_max=None):
    """
    Bayesian-style shrinkage toward global mean with tempered data weight:
       n_eff = min(n_i, n_max)  (if n_max is given)
       r_hat = (n_eff * mean_i + lam * mu) / (n_eff + lam)
    """
    n_raw = int(item_count.get(isbn, 0))
    m = float(item_mean.get(isbn, mu))
    n_eff = n_raw if n_max is None else min(n_raw, int(n_max))
    denom = n_eff + lam
    return (n_eff * m + lam * mu) / denom if denom > 0 else mu


# ===== Build a normalized title <-> ISBN map (for LLM lists) =====
def build_title_isbn_maps(books_df: pd.DataFrame):
    """
    Returns two dicts:
      title2isbn: normalized title -> a representative ISBN present in BX
      isbn2title: ISBN -> Book-Title (first occurrence)
    """
    isbn2title = {}
    title2isbn = {}
    for _, row in books_df[['ISBN', 'Book-Title']].iterrows():
        isbn = str(row['ISBN'])
        title = str(row['Book-Title'])
        normt = _norm_title(title)
        if isbn not in isbn2title:
            isbn2title[isbn] = title
        # take the first seen ISBN for this normalized title
        if normt and normt not in title2isbn:
            title2isbn[normt] = isbn
    return title2isbn, isbn2title

# --- One canonical mapper: titles -> ISBNs (dedup + normalization) ---
def titles_to_isbns(titles: list[str], title2isbn: dict[str, str]) -> list[str]:
    """Map human titles to dataset ISBNs using normalized keys; drop unknown/duplicates."""
    out, seen = [], set()
    for t in titles:
        k = _norm_title(t)
        isbn = title2isbn.get(k)
        if isbn and isbn not in seen:
            seen.add(isbn)
            out.append(isbn)
    return out

# ===== Compute group-aggregated graded relevance (oracle) =====

def build_oracle_for_group(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_users: list[int],
    strategy: str,
    books_df: pd.DataFrame,
    lam: int = LAMBDA_SHRINK,
    min_group_test: int = 1,
    n_max: int | None = None,
):
    """
    Oracle based ONLY on this group's TEST items (optional coverage rule).
    Missing member ratings can be imputed from TRAIN with tempered shrinkage.

    Returns:
      gain_by_isbn: dict(ISBN -> gain in [0,1])
      oracle_sorted_isbns: list of ISBNs by decreasing gain
    """

    # ---------- TRAIN stats for (optional) imputation ----------
    item_mean, item_count, mu = compute_item_stats(train_df)

    # ---------- Build TEST lookups FIRST (so closures can use them) ----------
    # Keep ISBNs as strings to match books_df maps
    test_rows = test_df[test_df['User-ID'].isin(group_users)][['User-ID', 'ISBN', 'Book-Rating']].copy()
    test_rows['ISBN'] = test_rows['ISBN'].astype(str)

    test_by_user: dict[int, dict[str, float]] = {u: {} for u in group_users}
    for _, r in test_rows.iterrows():
        uid = int(r['User-ID'])
        test_by_user[uid][str(r['ISBN'])] = float(r['Book-Rating'])
        # Keep only TEST items that exist in the catalog (have titles in books_df)
    catalog = set(books_df['ISBN'].astype(str))

    # ---------- Candidate pool: TEST items rated by the group (coverage rule) ----------
    # Union of items the group touched in TEST:
    candidate_isbns_all: set[str] = set()
    for u in group_users:
        candidate_isbns_all.update(test_by_user[u].keys())

    # Helper uses test_by_user which now EXISTS in enclosing scope
    def group_test_coverage(isbn: str) -> int:
        return sum(1 for u in group_users if isbn in test_by_user[u])

    candidate_isbns = [
        isbn for isbn in candidate_isbns_all
        if (group_test_coverage(isbn) >= min_group_test) and (isbn in catalog)
    ]

    # ---------- Aggregate to group gain ----------
    gain_by_isbn: dict[str, float] = {}
    for isbn in candidate_isbns:
        ratings = []
        for u in group_users:
            if isbn in test_by_user[u]:
                ratings.append(test_by_user[u][isbn])  # observed TEST label
            else:
                # tempered Bayesian imputation from TRAIN stats
                ratings.append(impute_item_shrink(isbn, item_mean, item_count, mu, lam=lam, n_max=n_max))

        if strategy == 'least_misery':
            g = min(ratings)
        elif strategy == 'happy_scenario':
            g = max(ratings)
        else:  # 'average'
            g = sum(ratings) / len(ratings)

        gain_by_isbn[isbn] = max(0.0, min(1.0, g / 10.0))  # linear gain in [0,1]

    oracle_sorted_isbns = sorted(gain_by_isbn.keys(), key=lambda x: gain_by_isbn[x], reverse=True)
    return gain_by_isbn, oracle_sorted_isbns, set(candidate_isbns)




# --- Recommendation Logic (Slightly modified for evaluation) ---

def get_cf_recommendations(algo, user_ids, ratings_df, books_df, n=5, strategy='average', k=100, exclude_isbns=[], allowed_isbns=None):
    """Group CF: aggregate per-item predictions; optionally restrict to allowed_isbns (TEST pool)."""
    candidate_items = set()

    # Catalog = intersection of ratings & books table
    books_in_catalog = set(books_df['ISBN'].astype(str))
    all_books = set(ratings_df['ISBN'].astype(str)) & books_in_catalog

    # NEW: if allowed_isbns (TEST pool) is given, restrict to it
    if allowed_isbns is not None:
        all_books = all_books & set(map(str, allowed_isbns))

    for user_id in user_ids:
        rated_books = set(ratings_df[ratings_df['User-ID'] == user_id]['ISBN'].astype(str))
        books_to_predict = list(all_books - rated_books - set(map(str, exclude_isbns)))
        # predict for a larger candidate set (k) per user, then union them
        preds = [algo.predict(user_id, isbn) for isbn in books_to_predict]
        top_k = sorted(preds, key=lambda x: x.est, reverse=True)[:k]
        for pred in top_k:
            candidate_items.add(str(pred.iid))

    # aggregate to group using the chosen strategy
    item_predictions = defaultdict(list)
    for item_id in candidate_items:
        for user_id in user_ids:
            est = algo.predict(user_id, item_id).est
            item_predictions[item_id].append(float(est))  # <- force plain float


    aggregated = []
    for item_id, vals in item_predictions.items():
        if len(vals) == len(user_ids):
            # vals are already float from the step above
            if strategy == 'least_misery':
                agg = min(vals)
            elif strategy == 'happy_scenario':
                agg = max(vals)
            else:
                agg = fmean(vals)          # <- robust to mixed numeric sources
            aggregated.append((item_id, float(agg)))


    aggregated.sort(key=lambda x: x[1], reverse=True)
    return [isbn for isbn, _ in aggregated[:n]]


def get_llm_recommendation(prompt: str) -> str:
    """Calls the OpenAI API with a cached client."""
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert book recommender. Your goal is to select the best books for a group from a list, based on a strategy."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"


def create_llm_prompt(strategy, user_profiles, candidate_titles=[], n=10):
    """Creates a prompt for the LLM to get top-N recommendations."""
    prompt = f"Here are the profiles of the users in the group:\n"
    for i, profile in enumerate(user_profiles):
        prompt += f"- User {i+1} loves: {', '.join(profile['likes'])}\n"
    
    if candidate_titles:
        prompt += f"\nFrom the following pre-selected list of books, which are the best {n} choices for the group?\n"
        prompt += "- " + "\n- ".join(candidate_titles)
    
    prompt += f"\n\nYour task is to recommend the top {n} best books for this group using the '{strategy}' strategy.\n"
    prompt += f"Please provide ONLY a numbered list of the top {n} book titles. Do not include justifications or any other text."
    return prompt

# --- Evaluation Logic ---

def find_ground_truth(ratings_df: pd.DataFrame, group_size: int = 3):
    """
    Finds a test case by first picking a book with enough ratings,
    then sampling 'group_size' users who rated it > 6.
    """
    # First ensure the book has at least 'group_size' total ratings
    isbn_counts = ratings_df['ISBN'].value_counts()
    viable_isbns = isbn_counts[isbn_counts >= group_size].index.tolist()

    random.shuffle(viable_isbns)

    for isbn in viable_isbns:
        # Users who rated this book > 6
        potential_users = ratings_df[(ratings_df['ISBN'] == isbn) & (ratings_df['Book-Rating'] > 6)]['User-ID']
        if len(potential_users) >= group_size:
            group_users = random.sample(list(potential_users), group_size)  # sample k distinct users
            return group_users, isbn

    return None, None


# --- Streamlit App Layout ---

st.title("📊 Recommender System Evaluation Demo")
#st.write("This app demonstrates the performance of three different group recommendation models by testing them against a known 'correct' answer.")
st.write("This app compares three group recommenders against an **Oracle (ideal) Top-K** list built from held-out TEST ratings.")

st.sidebar.title("⚙️ Controls")
strategy = st.sidebar.selectbox(
    "Choose a Group Recommendation Strategy:",
    ('average', 'least_misery', 'happy_scenario'),
    help="""
- **average**: Aims for overall group satisfaction.
- **least_misery**: Avoids books any member might dislike.
- **happy_scenario**: Focuses on books at least one member will love.
"""
)
use_memory_only = st.sidebar.checkbox(
    "Use RAG memory for user prefs (no CSV fallback)",
    value=True
)
group_size = st.sidebar.selectbox(
    "Group size",
    [3, 5, 10],
    index=0,
    help="How many users to include in the test group."
)

# --- Top-K control (1–20), default 10 ---
selected_k = st.sidebar.number_input(
    "Top-K (1–20)",
    min_value=1, max_value=20, value=10, step=1,
    help="All models will recommend K books; nDCG is computed at this K."
)

# --- Oracle coverage control ---
min_group_test = st.sidebar.slider(
    "Oracle: min TEST ratings per item (group)",
    min_value=1, max_value=3, value=1, step=1,
    help="Require an item be rated by at least this many group members in TEST to enter the Oracle."
)

# --- Temper popularity in Oracle imputation (cap effective support) ---
n_max = st.sidebar.number_input(
    "Oracle imputation cap n_max",
    min_value=10, max_value=2000, value=50, step=10,
    help="Caps the effective TRAIN support n_i when imputing missing TEST ratings inside the Oracle."
)


# --- Seed control: changes the sampled group & the train/test split ---
seed_val = st.sidebar.number_input(
    "Random seed (group & split)",
    min_value=0, max_value=1_000_000, value=RNG_SEED, step=1,
    help="Change this to reshuffle the sampled group and the train/test split."
)


algo = load_cf_model()
ratings_df, books_df = load_data()
ratings_df['ISBN'] = ratings_df['ISBN'].astype(str)
books_df['ISBN']   = books_df['ISBN'].astype(str)

# quick dataset stats
n_users   = int(ratings_df['User-ID'].nunique())
n_books   = int(books_df['ISBN'].nunique())
n_ratings = int(len(ratings_df))
st.sidebar.caption(f"Dataset: {n_users:,} users · {n_books:,} books · {n_ratings:,} ratings")


if algo and ratings_df is not None and books_df is not None:
    if st.button("🚀 Run Evaluation Demo"):

        with st.spinner("Finding a test group and building oracle..."):
            random.seed(seed_val)        # controls Python's RNG
            group_users, _ = find_ground_truth(ratings_df, group_size=group_size)

        if not group_users:
            st.error("Could not find a suitable test group. Please try again.")
        else:
           # --- Display run context ---
            st.info(f"**Strategy:** `{strategy}` | **Group size:** {group_size} | **Users:** {group_users}")
            train_df, test_df = split_ratings_train_test_for_group(
                ratings_df, group_users, test_frac=0.2, seed=seed_val
            )
            st.caption(
                f"Train/Test split per group member → TRAIN: {len(train_df)} rows, TEST: {len(test_df)} rows (seed={seed_val})."
            )

            # --- Build title<->ISBN maps once ---
            title2isbn, isbn2title = build_title_isbn_maps(books_df)

            # --- Build Oracle (ideal ranking) from TEST labels + shrinkage (TRAIN stats) ---
            gain_by_isbn, oracle_sorted_isbns, oracle_pool = build_oracle_for_group(
            train_df, test_df, group_users, strategy, books_df,
            min_group_test=min_group_test, n_max=n_max
            )

            # --- Show the Oracle Top-K wanted list ---
            oracle_topk_isbns  = oracle_sorted_isbns[:selected_k]
            oracle_topk_titles = [isbn2title.get(isbn, f"[Unknown ISBN {isbn}]") for isbn in oracle_topk_isbns]
            st.subheader(f"🎯 Oracle (Ideal) Top-{selected_k} for this group")
            st.markdown("\n".join([f"{i+1}. {t}" for i, t in enumerate(oracle_topk_titles)]))
            # Show which seed built this oracle
            st.caption(f"Oracle built with seed={seed_val}. Change it in the sidebar to reshuffle.")


            # --- Build User Profiles (TRAIN-only; filter out TEST titles to avoid leakage) ---
            user_profiles = []
            missing_users = []

            for user_id in group_users:
                likes = None

                # 1) Memory-first when checkbox is ON (comes from your seeded KV/FAISS)
                if use_memory_only:
                    likes = get_prefs_exact(str(user_id))  # list[str] or None

                # 2) If memory is ON but empty, keep demo flowing and record it
                if use_memory_only and not likes:
                    missing_users.append(user_id)
                    likes = ["General Fiction"]

                # 3) If checkbox is OFF, build likes from CSV (TRAIN-ish fallback)
                if not use_memory_only:
                    top_rated_isbns = ratings_df[
                        (ratings_df['User-ID'] == user_id) &
                        (ratings_df['Book-Rating'] > 7)
                    ].nlargest(5, 'Book-Rating')['ISBN']

                    likes = books_df[books_df['ISBN'].isin(top_rated_isbns)]['Book-Title'].tolist()
                    if not likes:
                        likes = ["General Fiction"]

                # 4) (Optional) If memory is ON but likes are very short, enrich via FAISS neighbors
                if use_memory_only and len(likes) < 3:
                    sims = retrieve_similar_prefs(likes, top_k=5)
                    if sims:
                        extra = [t.strip() for t in sims[0]['text'].split(";") if t.strip()]
                        for t in extra:
                            if t not in likes and len(likes) < 5:
                                likes.append(t)

                # 5) NEW: remove this user's TEST titles from likes (prevents label leakage in prompts)
                #    We map the user's TEST ISBNs to titles, normalize, then drop matches from likes.
                user_test_isbns = set(test_df[test_df['User-ID'] == user_id]['ISBN'])
                user_test_titles_norm = set()
                for isbn in user_test_isbns:
                    row = books_df[books_df['ISBN'] == isbn]
                    if not row.empty:
                        user_test_titles_norm.add(_norm_title(row['Book-Title'].iloc[0]))

                filtered_likes = [t for t in likes if _norm_title(t) not in user_test_titles_norm]
                # Keep original likes if filtering would empty the list (so the LLM still gets context)
                likes = filtered_likes or likes
                # keep the prompt compact
                likes = likes[:MAX_LIKES_PER_USER]

                user_profiles.append({'likes': likes})


            # Small heads-up if some users had no seeded memory
            if use_memory_only and missing_users:
                st.warning(f"⚠️ Memory missing for users: {missing_users[:5]}{'...' if len(missing_users)>5 else ''}")

            evaluation_results = []
            with st.spinner(f"Running models with '{strategy}' strategy..."):
               # --- 1. Pure CF Model (restricted to TEST pool) ---
                cf_recs_isbns = get_cf_recommendations(
                algo, group_users, ratings_df, books_df,
                n=50, strategy=strategy,
                exclude_isbns=[],
                allowed_isbns=oracle_pool       # <- use allowed_isbns, not candidate_pool
                )
                cf_titles = [books_df[books_df['ISBN'] == isbn]['Book-Title'].iloc[0] for isbn in cf_recs_isbns]

                # --- 2. Pure LLM Model (restricted to TEST pool) ---
                candidate_titles_from_test = (
                    [isbn2title[i] for i in oracle_pool if i in isbn2title][:MAX_TITLES_FOR_LLM]
                )

                llm_prompt = create_llm_prompt(
                    strategy,
                    user_profiles,
                    candidate_titles=candidate_titles_from_test,  # <- only TEST titles
                    n=selected_k
                )
                llm_rec = get_llm_recommendation(llm_prompt)

                # --- 3. Hybrid Model (CF shortlist from TEST pool → LLM re-rank) ---
                stage1_isbns = get_cf_recommendations(
                algo, group_users, ratings_df, books_df,
                n=min(50, len(oracle_pool)), strategy=strategy,
                exclude_isbns=[],
                allowed_isbns=oracle_pool
                 )

                random.shuffle(stage1_isbns)
                stage1_titles = [isbn2title[i] for i in stage1_isbns if i in isbn2title][:MAX_TITLES_FOR_LLM]

                hybrid_prompt = create_llm_prompt(
                    strategy,
                    user_profiles,
                    candidate_titles=stage1_titles,               # <- LLM only sees TEST-pool shortlist
                    n=selected_k
                )
                hybrid_rec = get_llm_recommendation(hybrid_prompt)

                # ===== Oracle-based graded nDCG@K =====

                llm_list    = extract_titles_from_llm_output(llm_rec)[:selected_k]
                hybrid_list = extract_titles_from_llm_output(hybrid_rec)[:selected_k]

                llm_isbns    = titles_to_isbns(llm_list, title2isbn)
                hybrid_isbns = titles_to_isbns(hybrid_list, title2isbn)
                cf_isbns     = list(map(str, cf_recs_isbns))     

                oracle_topk_isbns = list(map(str, oracle_sorted_isbns[:selected_k]))
 
                # ===== Binary top-K nDCG vs Oracle Top-K =====
                ndcg_bin_cf     = ndcg_at_k_binary(cf_isbns,     oracle_topk_isbns, selected_k)
                ndcg_bin_llm    = ndcg_at_k_binary(llm_isbns,    oracle_topk_isbns, selected_k)
                ndcg_bin_hybrid = ndcg_at_k_binary(hybrid_isbns, oracle_topk_isbns, selected_k)

                # (optional) keep graded nDCG if you still want to compute/show it later
                ndcg_cf     = ndcg_at_k_graded(cf_isbns,     gain_by_isbn, k=selected_k)
                ndcg_llm    = ndcg_at_k_graded(llm_isbns,    gain_by_isbn, k=selected_k)
                ndcg_hybrid = ndcg_at_k_graded(hybrid_isbns, gain_by_isbn, k=selected_k)

                # ----- Binary nDCG banner + trophies (guard all-zero case) -----
                all_zero_bin = (ndcg_bin_cf == 0.0 and ndcg_bin_llm == 0.0 and ndcg_bin_hybrid == 0.0)

                trophies = ["", "", ""]  # row order: [CF, LLM, Hybrid]

                if all_zero_bin:
                    st.info(f"No overlap with Oracle Top-{selected_k}; all Binary nDCG@{selected_k} = 0.0. Skipping hypothesis banner.")
                else:
                    best = max(ndcg_bin_cf, ndcg_bin_llm, ndcg_bin_hybrid)

                    # mark winners for the table
                    winners = []
                    if abs(ndcg_bin_cf - best) < 1e-12:     winners.append(0)
                    if abs(ndcg_bin_llm - best) < 1e-12:    winners.append(1)
                    if abs(ndcg_bin_hybrid - best) < 1e-12: winners.append(2)
                    trophies = ["✅" if i in winners else "" for i in range(3)]

                    # show hypothesis banner keyed to Binary nDCG
                    if (ndcg_bin_hybrid > ndcg_bin_cf) and (ndcg_bin_hybrid > ndcg_bin_llm):
                        st.success(
                            f"✔ Hypothesis supported: Hybrid is best by Binary Oracle nDCG@{selected_k} "
                            f"(Hybrid {ndcg_bin_hybrid:.3f} vs CF {ndcg_bin_cf:.3f}, LLM {ndcg_bin_llm:.3f})."
                        )
                    elif abs(ndcg_bin_hybrid - best) < 1e-12:
                        st.warning(
                            f"≈ Hypothesis partially supported (tie) at Binary Oracle nDCG@{selected_k}. "
                            f"Hybrid {ndcg_bin_hybrid:.3f}, CF {ndcg_bin_cf:.3f}, LLM {ndcg_bin_llm:.3f}."
                        )
                    else:
                        st.error(
                            f"✖ Hypothesis not supported: Hybrid below best at Binary Oracle nDCG@{selected_k}. "
                            f"Hybrid {ndcg_bin_hybrid:.3f}, CF {ndcg_bin_cf:.3f}, LLM {ndcg_bin_llm:.3f}."
                        )

                # ===== end Oracle-based graded nDCG@K =====

            # --- Display Results ---
            st.subheader("Evaluation Results")
            
            # Format recommendations as a numbered list string
            cf_display = "\n".join([f"{i+1}. {title}" for i, title in enumerate(cf_titles)])
            
            results_data = {
                "Model": ["Pure Collaborative Filtering", "Pure LLM", "Hybrid (CF + LLM)"],
                f"Top {selected_k} Recommendations": [cf_display, llm_rec, hybrid_rec],
            }


            df_results = pd.DataFrame(results_data)
            # ===== Conditionally add Oracle nDCG@K columns =====
            # Always show the metric column, even if it’s all zeros
            df_results[f"Binary Oracle nDCG@{selected_k}"] = [
                f"{ndcg_bin_cf:.4f}",
                f"{ndcg_bin_llm:.4f}",
                f"{ndcg_bin_hybrid:.4f}",
            ]

            # Only add the “Best by …” trophy column when we actually have a non-zero best
            if not all_zero_bin:
                df_results[f"Best by Oracle nDCG@{selected_k}"] = trophies
            # ===== end Oracle nDCG@K columns =====

            st.dataframe(df_results.style.set_properties(**{'white-space': 'pre-wrap'}))

else:
    st.info("Waiting for data and models to be loaded...")
