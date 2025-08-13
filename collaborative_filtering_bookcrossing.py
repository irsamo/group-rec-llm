import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy
import pickle
import os
from collections import Counter, defaultdict
from statistics import mean

# Load the BookCrossing ratings dataset
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1')

# Prepare the data for Surprise
reader = Reader(rating_scale=(ratings['Book-Rating'].min(), ratings['Book-Rating'].max()))
data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)

# --- Model Training/Loading ---
model_path = 'svd_model.pkl'
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

if os.path.exists(model_path):
    print("Loading saved model from svd_model.pkl...")
    with open(model_path, 'rb') as f:
        algo = pickle.load(f)
    print("Model loaded.")
else:
    print("No saved model found. Training a new one...")
    # --- Hyperparameter Tuning with GridSearchCV ---
    param_grid = {
        "n_epochs": [20, 30],      # Number of iterations
        "lr_all": [0.005, 0.01],    # Learning rate
        "reg_all": [0.02, 0.1],     # Regularization term
        "n_factors": [50, 100]      # Number of factors
    }
    print("Performing grid search... (This may take a while with your large dataset)")
    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, n_jobs=-1)
    gs.fit(data)

    # Get the best SVD algorithm
    algo = gs.best_estimator["rmse"]
    print(f"Best cross-validation RMSE: {gs.best_score['rmse']:.4f}")
    print("Best parameters found:", gs.best_params["rmse"])

    # Train the best model on the full training set
    print("Training the best model on the full training set...")
    algo.fit(trainset)
    
    # Save the trained model
    print(f"Saving trained model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(algo, f)

# Predict ratings for the test set
print("Evaluating model on the test set...")
predictions = algo.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)
print(f"Test RMSE: {rmse:.4f}")

# Example: Recommend top 5 books for a given user
def get_top_n_recommendations_from_test(algo, user_id, ratings, testset, n=5):
    # Filter testset for this user
    user_test_items = [(uid, iid, true_r) for (uid, iid, true_r) in testset if uid == user_id]
    # Predict ratings for these items
    predictions = [algo.predict(user_id, iid) for (_, iid, _) in user_test_items]
    # Attach actual ratings
    pred_with_actual = [
        (pred.iid, pred.est, true_r)
        for pred, (_, _, true_r) in zip(predictions, user_test_items)
    ]
    # Sort by predicted rating
    top_n = sorted(pred_with_actual, key=lambda x: x[1], reverse=True)[:n]
    return top_n

# --- Group Recommendation Logic ---
def get_group_recommendations(algo, user_ids, ratings, n=10, strategy='average', k=100):
    """
    Generates group recommendations for a list of users using a specified strategy.

    Args:
        algo: The trained Surprise algorithm.
        user_ids (list): A list of user IDs for the group.
        ratings (pd.DataFrame): The original ratings dataframe.
        n (int): The number of recommendations to return.
        strategy (str): The aggregation strategy ('average', 'least_misery', or 'happy_scenario').
        k (int): The number of top recommendations per user to consider for the candidate pool.

    Returns:
        list: A list of tuples containing (ISBN, aggregated_score).
    """
    # Step 1: Generate top-k recommendations for each user to create a candidate pool
    candidate_items = set()
    all_books = set(ratings['ISBN'])
    for user_id in user_ids:
        # Get books not yet rated by the user
        rated_books = set(ratings[ratings['User-ID'] == user_id]['ISBN'])
        books_to_predict = list(all_books - rated_books)
        
        # Predict ratings for unrated books
        predictions = [algo.predict(user_id, isbn) for isbn in books_to_predict]
        
        # Get the top k predictions and add them to the candidate pool
        top_k = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]
        for pred in top_k:
            candidate_items.add(pred.iid)

    # Step 2: Get predictions for all users on the candidate items
    item_predictions = defaultdict(list)
    for item_id in candidate_items:
        for user_id in user_ids:
            pred = algo.predict(user_id, item_id)
            item_predictions[item_id].append(pred.est)

    # Step 3: Aggregate the predictions based on the chosen strategy
    aggregated_recommendations = []
    for item_id, preds in item_predictions.items():
        # Ensure we have a prediction from every user in the group
        if len(preds) == len(user_ids):
            agg_score = 0
            if strategy == 'average':
                agg_score = mean([float(p) for p in preds])
            elif strategy == 'least_misery':
                agg_score = min(preds)
            elif strategy == 'happy_scenario':
                agg_score = max(preds)
            aggregated_recommendations.append((item_id, agg_score))

    # Step 4: Sort the aggregated recommendations and return the top n
    aggregated_recommendations.sort(key=lambda x: x[1], reverse=True)
    return aggregated_recommendations[:n]

# Example usage
# Find a user in the test set with at least one rating
test_users = [uid for (uid, _, _) in testset]
user_counts = Counter(test_users)
user_id = None
for uid, count in user_counts.items():
    if count > 0:
        user_id = uid
        break

if user_id is not None:
    recommendations = get_top_n_recommendations_from_test(algo, user_id, ratings, testset, n=5)
    print(f"Top 5 recommendations for user {user_id} (from test set):")
    for isbn, est_rating, actual_rating in recommendations:
        print(f"ISBN: {isbn}, Predicted Rating: {est_rating:.2f}, Actual Rating: {actual_rating}")
else:
    print("No user with ratings in the test set was found.")

# --- Example of Group Recommendations ---
# Find a few users from the test set to form a group
group_user_ids = []
if user_counts:
    # Get the 3 users with the most ratings in the test set
    top_users = user_counts.most_common(3)
    if len(top_users) == 3:
        group_user_ids = [uid for uid, count in top_users]

if group_user_ids:
    print(f"\n--- Generating Group Recommendations for users: {group_user_ids} ---")

    # 1. Average Strategy
    avg_recs = get_group_recommendations(algo, group_user_ids, ratings, n=5, strategy='average')
    print("\nTop 5 Group Recommendations (Average Strategy):")
    for isbn, score in avg_recs:
        print(f"ISBN: {isbn}, Average Predicted Score: {score:.2f}")

    # 2. Least Misery Strategy
    lm_recs = get_group_recommendations(algo, group_user_ids, ratings, n=5, strategy='least_misery')
    print("\nTop 5 Group Recommendations (Least Misery Strategy):")
    for isbn, score in lm_recs:
        print(f"ISBN: {isbn}, Minimum Predicted Score: {score:.2f}")
    
    # 3. Happy Scenario Strategy
    hs_recs = get_group_recommendations(algo, group_user_ids, ratings, n=5, strategy='happy_scenario')
    print("\nTop 5 Group Recommendations (Happy Scenario Strategy):")
    for isbn, score in hs_recs:
        print(f"ISBN: {isbn}, Maximum Predicted Score: {score:.2f}")
else:
    print("\nCould not form a group of 3 users from the test set.")
