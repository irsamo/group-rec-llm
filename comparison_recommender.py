import os
import pandas as pd
import pickle
from collections import Counter, defaultdict
from statistics import mean
from openai import OpenAI
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# --- 1. Load All Necessary Data and Models ---

def load_cf_model(model_path='svd_model.pkl'):
    """Loads the pre-trained collaborative filtering model."""
    if not os.path.exists(model_path):
        print("Error: Saved model 'svd_model.pkl' not found. Please run the main script first to train and save the model.")
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def load_data():
    """Loads the ratings and books datasets."""
    try:
        ratings_df = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        books_df = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        return ratings_df, books_df
    except FileNotFoundError as e:
        print(f"Error: Could not find a required data file. {e}")
        return None, None

# --- 2. Collaborative Filtering (Quantitative) Group Logic ---

def get_cf_group_recommendations(algo, user_ids, ratings_df, books_df, n=1, strategy='average', k=100):
    """Generates a single group recommendation using the CF model."""
    candidate_items = set()
    all_books = set(ratings_df['ISBN'])
    for user_id in user_ids:
        rated_books = set(ratings_df[ratings_df['User-ID'] == user_id]['ISBN'])
        books_to_predict = list(all_books - rated_books)
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
            agg_score = 0
            if strategy == 'average':
                agg_score = mean([float(p) for p in preds])
            elif strategy == 'least_misery':
                agg_score = min(preds)
            elif strategy == 'happy_scenario':
                agg_score = max(preds)
            aggregated_recommendations.append((item_id, agg_score))

    aggregated_recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top recommendation and its title
    if not aggregated_recommendations:
        return "No recommendation found", 0.0
        
    top_isbn = aggregated_recommendations[0][0]
    top_score = aggregated_recommendations[0][1]
    book_title = books_df.loc[books_df['ISBN'] == top_isbn, 'Book-Title'].iloc[0]
    
    return f"'{book_title}'", f"{top_score:.2f}"


# --- 3. LLM (Qualitative) Group Logic ---

def get_llm_recommendation(prompt):
    """Calls the OpenAI API to get a book recommendation."""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not set."
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert book recommender for groups. Your goal is to suggest a single book that fits the group's collective taste based on a specific strategy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

def create_llm_prompt(strategy, user_profiles):
    """Creates a detailed prompt for the LLM."""
    prompt = f"Here are the profiles of the 3 users in the group, based on their top-rated books:\n"
    for i, profile in enumerate(user_profiles):
        prompt += f"- User {i+1} loves: {', '.join(profile['likes'])}\n"
    
    prompt += f"\nYour task is to recommend a single book for the group using the '{strategy}' strategy.\n"

    if strategy == 'average':
        prompt += "This means you should find a book that everyone in the group is likely to find enjoyable. Aim for a solid, crowd-pleasing choice."
    elif strategy == 'least_misery':
        prompt += "This means you must prioritize finding a book that no one will dislike. It's better to recommend a safe book than a risky one."
    elif strategy == 'happy_scenario':
        prompt += "This means you should find a book that at least one person will absolutely love, even if others are just neutral."

    prompt += "\n\nPlease provide ONLY the book title and a one-sentence justification for your choice."
    return prompt


# --- 4. Main Comparison Logic ---

def main():
    # Load models and data
    algo = load_cf_model()
    ratings_df, books_df = load_data()
    if algo is None or ratings_df is None or books_df is None:
        return

    # Find a group of 3 users
    _, testset = train_test_split(Dataset.load_from_df(ratings_df[['User-ID', 'ISBN', 'Book-Rating']], Reader()), test_size=0.2, random_state=42)
    user_counts = Counter(uid for uid, _, _ in testset)
    top_users = user_counts.most_common(3)
    if len(top_users) < 3:
        print("Could not find 3 users in the test set to form a group.")
        return
    group_user_ids = [uid for uid, count in top_users]
    print(f"--- Comparing recommendations for group: {group_user_ids} ---")

    # Build rich user profiles for the LLM
    user_profiles = []
    for user_id in group_user_ids:
        top_rated_isbns = ratings_df[(ratings_df['User-ID'] == user_id) & (ratings_df['Book-Rating'] > 7)].nlargest(5, 'Book-Rating')['ISBN']
        top_titles = books_df[books_df['ISBN'].isin(top_rated_isbns)]['Book-Title'].tolist()
        user_profiles.append({'likes': top_titles if top_titles else ["General Fiction"]})

    strategies = ['average', 'least_misery', 'happy_scenario']
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.replace('_', ' ').title()} ---")
        
        # Get Quantitative Recommendation
        cf_title, cf_score = get_cf_group_recommendations(algo, group_user_ids, ratings_df, books_df, strategy=strategy)
        print(f"Quantitative (CF) Rec: {cf_title} (Score: {cf_score})")

        # Get Qualitative Recommendation
        llm_prompt = create_llm_prompt(strategy, user_profiles)
        llm_rec = get_llm_recommendation(llm_prompt)
        print(f"Qualitative (LLM) Rec: {llm_rec}")

if __name__ == "__main__":
    main()
