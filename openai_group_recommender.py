import os
import pandas as pd
from openai import OpenAI

def get_llm_recommendation(prompt):
    """
    Calls the OpenAI API to get a book recommendation based on a prompt.
    """
    try:
        # Ensure the API key is set
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY environment variable not set."

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert book recommender for groups. Your goal is to suggest a single book that fits the group's collective taste based on a specific strategy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

def create_prompt(strategy, user_profiles, book_list):
    """
    Creates a detailed prompt for the LLM based on the recommendation strategy.
    """
    prompt = f"Here are the profiles of the 3 users in the group:\n"
    for i, profile in enumerate(user_profiles):
        prompt += f"- User {i+1} likes: {', '.join(profile['likes'])}\n"
    
    prompt += "\nHere is a list of available books they could be recommended:\n"
    prompt += ", ".join(book_list)
    
    prompt += f"\n\nYour task is to recommend a single book for the group using the '{strategy}' strategy.\n"

    if strategy == 'average':
        prompt += "This means you should find a book that everyone in the group is likely to find enjoyable, even if it's not anyone's absolute favorite. Aim for a solid, crowd-pleasing choice."
    elif strategy == 'least_misery':
        prompt += "This means you must prioritize finding a book that no one in the group will dislike. It's better to recommend a safe book that everyone finds acceptable than a risky book that one person might hate."
    elif strategy == 'happy_scenario':
        prompt += "This means you should find a book that at least one person in the group will absolutely love, even if the others are just neutral about it. Aim for a book that could become someone's new favorite."

    prompt += "\n\nPlease provide the book title and a brief justification for your choice based on the strategy."
    return prompt

def main():
    # --- Load Book Data ---
    try:
        books_df = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip')
        # Get a sample of book titles to use in the prompt
        clean_titles = books_df['Book-Title'].dropna()
        sample_size = min(100, len(clean_titles))
        book_titles = clean_titles.sample(n=sample_size, random_state=42).tolist()
    except FileNotFoundError:
        print("Error: BX-Books.csv not found. Please make sure the file is in the correct directory.")
        return

    # --- Define User Profiles ---
    user_profiles = [
        {"likes": ["The Hobbit", "Dune", "Neuromancer"]},
        {"likes": ["Pride and Prejudice", "Where the Crawdads Sing", "The Nightingale"]},
        {"likes": ["The Silent Patient", "Gone Girl", "The Da Vinci Code"]}
    ]

    # --- Generate Recommendations for each strategy ---
    strategies = ['average', 'least_misery', 'happy_scenario']
    
    for strategy in strategies:
        print(f"\n--- Generating Recommendation for Strategy: {strategy.replace('_', ' ').title()} ---")
        prompt = create_prompt(strategy, user_profiles, book_titles)
        recommendation = get_llm_recommendation(prompt)
        print(recommendation)

if __name__ == "__main__":
    main()
