import streamlit as st
import pandas as pd
import pickle
import os
import re, math
import random
from collections import Counter, defaultdict
from statistics import mean
from openai import OpenAI
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from memory_store import get_prefs_exact, retrieve_similar_prefs

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

# --- Optional: rank helper (keep only if used elsewhere) ---
def rank_of_title_in_list(title_list, target_title):
    """Return 1-based rank of target_title (case-insensitive), else None."""
    tgt = _norm_title(target_title)
    for i, t in enumerate(title_list, start=1):
        if _norm_title(t) == tgt:
            return i
    return None

# --- nDCG@10 for a single relevant item (binary relevance) ---
def ndcg_at10_single_hit(ranked_titles: list[str], ground_truth_title: str) -> float:
    """
    If the relevant title appears at rank r (1-based) within the top-10:
        nDCG@10 = 1 / log2(r + 1)
    else 0. With a single relevant item, IDCG@10 = 1.
    """
    if not ranked_titles or not ground_truth_title:
        return 0.0

    gt = _norm_title(ground_truth_title)

    # normalize + de-duplicate (first occurrence counts)
    seen, ranked = set(), []
    for t in ranked_titles[:10]:
        nt = _norm_title(t)
        if nt and nt not in seen:
            seen.add(nt)
            ranked.append(nt)

    try:
        r = ranked.index(gt) + 1  # 1-based rank
        return 1.0 / math.log2(r + 1)
    except ValueError:
        return 0.0

def ndcg_single_hit_at_k(ranked_titles: list[str], ground_truth_title: str, k: int) -> float:
    """
    Single-relevant-item nDCG@K: 1/log2(r+1) if hit at rank r<=K; else 0.
    """
    if not ranked_titles or not ground_truth_title or k <= 0:
        return 0.0
    gt = _norm_title(ground_truth_title)
    seen, ranked = set(), []
    for t in ranked_titles[:k]:
        nt = _norm_title(t)
        if nt and nt not in seen:
            seen.add(nt)
            ranked.append(nt)
    try:
        r = ranked.index(gt) + 1
        return 1.0 / math.log2(r + 1)
    except ValueError:
        return 0.0


# ----------------------------------------------


# --- Recommendation Logic (Slightly modified for evaluation) ---

def get_cf_recommendations(algo, user_ids, ratings_df, books_df, n=5, strategy='average', k=100, exclude_isbns=[]):
    """Generates top-n group recommendations using the CF model, can exclude items."""
    candidate_items = set()
    all_books = set(ratings_df['ISBN'])
    for user_id in user_ids:
        rated_books = set(ratings_df[ratings_df['User-ID'] == user_id]['ISBN'])
        books_to_predict = list(all_books - rated_books - set(exclude_isbns))
        predictions = [algo.predict(user_id, isbn) for isbn in books_to_predict]
        top_k = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]
        for pred in top_k:
            candidate_items.add(pred.iid)

    item_predictions = defaultdict(list)
    for item_id in candidate_items:
        for user_id in user_ids:
            pred = algo.predict(user_id, item_id)
            item_predictions[item_id].append(pred.est)

    aggregated_recommendations = []
    for item_id, preds in item_predictions.items():
        if len(preds) == len(user_ids):
            agg_score = mean([float(p) for p in preds]) if strategy == 'average' else min(preds) if strategy == 'least_misery' else max(preds)
            aggregated_recommendations.append((item_id, agg_score))

    aggregated_recommendations.sort(key=lambda x: x[1], reverse=True)
    top_n_isbns = [isbn for isbn, score in aggregated_recommendations[:n]]
    return top_n_isbns

def get_llm_recommendation(prompt):
    """Calls the OpenAI API."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: return "Error: OPENAI_API_KEY not set."
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert book recommender. Your goal is to select the best books for a group from a list, based on a strategy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5, max_tokens=1000 # Increased tokens for longer list
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

def find_ground_truth(ratings_df):
    """
    Finds a ground truth test case by first finding a suitable book 
    and then forming a group from users who rated it highly.
    This is more robust for sparse datasets.
    """
    # Count how many ratings each book has
    isbn_counts = ratings_df['ISBN'].value_counts()
    
    # Filter for books that have been rated by at least 5 users
    # This increases the chance of finding a good test case.
    viable_isbns = isbn_counts[isbn_counts >= 5].index
    
    # Shuffle the viable ISBNs to ensure randomness in test cases
    shuffled_isbns = list(viable_isbns)
    random.shuffle(shuffled_isbns)
    
    for isbn in shuffled_isbns:
        # Find all users who rated this book with a score > 6
        potential_users = ratings_df[(ratings_df['ISBN'] == isbn) & (ratings_df['Book-Rating'] > 6)]['User-ID']
        
        if len(potential_users) >= 3:
            # We found a book rated highly by at least 3 people.
            # Form a group from them.
            group_users = random.sample(list(potential_users), 3)
            return group_users, isbn
            
    return None, None # Return None if no suitable book/group is found

# --- Streamlit App Layout ---

st.title("📊 Recommender System Evaluation Demo")
st.write("This app demonstrates the performance of three different group recommendation models by testing them against a known 'correct' answer.")

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
# --- Top-K control (1..20), default 10 ---
selected_k = st.sidebar.number_input(
    "Top-K (1–20)",
    min_value=1, max_value=20, value=10, step=1,
    help="All models will recommend K books; nDCG is computed at this K."
)


algo = load_cf_model()
ratings_df, books_df = load_data()

if algo and ratings_df is not None and books_df is not None:
    if st.button("🚀 Run Evaluation Demo"):
        with st.spinner("Finding a test group and ground truth..."):
            group_users, ground_truth_isbn = find_ground_truth(ratings_df)
        
        if not group_users:
            st.error("Could not find a suitable test group. Please try again.")
        else:
            ground_truth_title = books_df[books_df['ISBN'] == ground_truth_isbn]['Book-Title'].iloc[0]
            st.info(f"**Strategy:** `{strategy}` | **Evaluation Goal:** The models must recommend **'{ground_truth_title}'** for group `{group_users}`.")

            # --- Build User Profiles (excluding ground truth) ---
            user_profiles = []
            missing_users = []

            for user_id in group_users:
                likes = None

                # 1) Memory-first when checkbox is ON
                if use_memory_only:
                    likes = get_prefs_exact(str(user_id))  # returns list of titles or None

                # 2) If memory is ON but empty for this user, keep app running but note it
                if use_memory_only and not likes:
                    missing_users.append(user_id)
                    likes = ["General Fiction"]  # keeps the demo flowing

                # 3) If checkbox is OFF, use your original CSV-based extraction
                if not use_memory_only:
                    top_rated_isbns = ratings_df[
                        (ratings_df['User-ID'] == user_id) &
                        (ratings_df['Book-Rating'] > 7) &
                        (ratings_df['ISBN'] != ground_truth_isbn)
                    ].nlargest(5, 'Book-Rating')['ISBN']

                    likes = books_df[books_df['ISBN'].isin(top_rated_isbns)]['Book-Title'].tolist()
                    if not likes:
                        likes = ["General Fiction"]

                # 4) (Optional) If memory is ON but likes are very short, enrich from similar users
                if use_memory_only and len(likes) < 3:
                    sims = retrieve_similar_prefs(likes, top_k=5)
                    if sims:
                        extra = [t.strip() for t in sims[0]['text'].split(";") if t.strip()]
                        for t in extra:
                            if t not in likes and len(likes) < 5:
                                likes.append(t)

                user_profiles.append({'likes': likes})

            # Small heads-up if some users had no seeded memory
            if use_memory_only and missing_users:
                st.warning(f"⚠️ Memory missing for users: {missing_users[:5]}{'...' if len(missing_users)>5 else ''}")

            evaluation_results = []
            with st.spinner(f"Running models with '{strategy}' strategy..."):
                # --- 1. Pure CF Model ---
                cf_recs_isbns = get_cf_recommendations(
                    algo, group_users, ratings_df, books_df,
                    n=selected_k, strategy=strategy, exclude_isbns=[ground_truth_isbn]
                )
                cf_success = ground_truth_isbn in cf_recs_isbns
                cf_titles = [books_df[books_df['ISBN'] == isbn]['Book-Title'].iloc[0] for isbn in cf_recs_isbns if not books_df[books_df['ISBN'] == isbn].empty]
                
                # --- 2. Pure LLM Model ---
                llm_prompt = create_llm_prompt(strategy, user_profiles, n=selected_k)
                llm_rec = get_llm_recommendation(llm_prompt)
                llm_success = ground_truth_title.lower() in llm_rec.lower()

                # --- 3. Hybrid Model ---
                candidate_isbns = get_cf_recommendations(algo, group_users, ratings_df, books_df, n=50, strategy=strategy, exclude_isbns=[ground_truth_isbn]) # Generate more candidates
                # Artificially add the ground truth to the candidate list to test re-ranking
                if ground_truth_isbn not in candidate_isbns:
                    candidate_isbns.append(ground_truth_isbn)
                random.shuffle(candidate_isbns)
                candidate_titles = [books_df[books_df['ISBN'] == isbn]['Book-Title'].iloc[0] for isbn in candidate_isbns if not books_df[books_df['ISBN'] == isbn].empty]
                
                hybrid_prompt = create_llm_prompt(strategy, user_profiles, candidate_titles, n=selected_k)
                hybrid_rec = get_llm_recommendation(hybrid_prompt)
                hybrid_success = ground_truth_title.lower() in hybrid_rec.lower()
                # ===== NDCG@K scoring (single ground-truth title) =====
                # Build model outputs up to the user-selected K
                llm_top_k    = extract_titles_from_llm_output(llm_rec)[:selected_k]
                hybrid_top_k = extract_titles_from_llm_output(hybrid_rec)[:selected_k]
                cf_top_k     = [t for t in cf_titles if t][:selected_k]

                # Compute nDCG@K (binary, single relevant item)
                ndcg_cf     = ndcg_single_hit_at_k(cf_top_k,     ground_truth_title, k=selected_k)
                ndcg_llm    = ndcg_single_hit_at_k(llm_top_k,    ground_truth_title, k=selected_k)
                ndcg_hybrid = ndcg_single_hit_at_k(hybrid_top_k, ground_truth_title, k=selected_k)

                # Show NDCG column only if at least one model hits at K
                _show_ndcg = (ndcg_cf > 0.0) or (ndcg_llm > 0.0) or (ndcg_hybrid > 0.0)

                # Always initialize; avoids NameError on Streamlit reruns
                trophies = ["", "", ""]

                if _show_ndcg:
                    _scores  = {"CF": ndcg_cf, "LLM": ndcg_llm, "Hybrid": ndcg_hybrid}
                    _best    = max(_scores.values())
                    _winners = [name for name, val in _scores.items() if abs(val - _best) < 1e-12]
                    # Row order below is CF, LLM, Hybrid
                    order = ["CF", "LLM", "Hybrid"]
                    for i, key in enumerate(order):
                        if key in _winners:
                            trophies[i] = "✅"
                # ===== end NDCG@K scoring =====




            # --- Display Results ---
            st.subheader("Evaluation Results")
            
            # Format recommendations as a numbered list string
            cf_display = "\n".join([f"{i+1}. {title}" for i, title in enumerate(cf_titles)])
            
            results_data = {
                "Model": ["Pure Collaborative Filtering", "Pure LLM", "Hybrid (CF + LLM)"],
                f"Top {selected_k} Recommendations": [cf_display, llm_rec, hybrid_rec],
                "Found Correct Book?": [
                    "✅ Yes" if cf_success else "❌ No",
                    "✅ Yes" if llm_success else "❌ No",
                    "✅ Yes" if hybrid_success else "❌ No"
                ],
            }

            df_results = pd.DataFrame(results_data)
            # ===== Conditionally add NDCG@K columns =====
            if _show_ndcg:
                df_results[f"NDCG@{selected_k}"]         = [f"{ndcg_cf:.4f}", f"{ndcg_llm:.4f}", f"{ndcg_hybrid:.4f}"]
                df_results[f"Best by NDCG@{selected_k}"] = trophies
            # ===== end conditional NDCG@K columns =====



            
            st.dataframe(df_results.style.set_properties(**{'white-space': 'pre-wrap'}))

            if hybrid_success:
                st.success("### Hypothesis Confirmed\nThe Hybrid model successfully identified the correct book from the candidate list, demonstrating its ability to combine contextual understanding with data-driven insights—a task where the other models often fail.")
            elif cf_success or llm_success:
                st.warning("### Hypothesis Partially Met\nOne of the baseline models found the correct book, but the hybrid model did not. This can happen if the LLM re-ranking step incorrectly filters out the right answer.")
            else:
                st.error("### Hypothesis Not Confirmed\nNone of the models were able to find the correct book in this run. This highlights the challenges of group recommendation with sparse data.")
else:
    st.info("Waiting for data and models to be loaded...")
