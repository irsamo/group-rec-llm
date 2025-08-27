# seed_memory_from_dataset.py
import pandas as pd
from memory_store import upsert_user_prefs_batch

RATINGS_PATH = "BX-Book-Ratings.csv"
BOOKS_PATH   = "BX-Books.csv"

def main():
    # Read with Book-Crossing quirks
    ratings = pd.read_csv(RATINGS_PATH, sep=";", encoding="latin-1", on_bad_lines="skip")
    books   = pd.read_csv(BOOKS_PATH,   sep=";", encoding="latin-1", on_bad_lines="skip")

    # Filter explicit ratings only (Book-Rating > 0) and prefer strong likes
    ratings = ratings[ratings["Book-Rating"] > 0]
    strong  = ratings[ratings["Book-Rating"] >= 8]  # you can tune 7/8/9

    # Keep only ISBNs present in books (defensive)
    strong = strong[strong["ISBN"].isin(books["ISBN"])]

    # Map ISBN -> Title
    isbn2title = books.set_index("ISBN")["Book-Title"].fillna("").to_dict()

    # Build per-user top titles (up to N)
    N_PER_USER = 10
    records = []
    for uid, grp in strong.groupby("User-ID"):
        # top N by rating, then by count
        top = grp.sort_values(["Book-Rating"], ascending=False).head(N_PER_USER)
        titles = [isbn2title.get(isbn, "").strip() for isbn in top["ISBN"]]
        titles = [t for t in titles if t]  # drop empties
        if len(titles) >= 3:
            records.append({"user_id": str(uid), "prefs": titles})

    print(f"Seeding memory for {len(records)} users ...")
    upsert_user_prefs_batch(records, importance=3)
    print("Done. You should now have user_prefs.faiss / *_meta.json / *_kv.json.")

if __name__ == "__main__":
    main()
