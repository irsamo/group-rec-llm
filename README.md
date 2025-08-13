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

## ⚙️ How to Run the Application

1.  **Set up the Environment**:
    *   Ensure you have Python 3.8+ installed.
    *   Create and activate a virtual environment:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   Install the required packages:
        ```bash
        pip install pandas "numpy<2.0" scikit-surprise openai streamlit
        ```

2.  **Set the OpenAI API Key**:
    *   The LLM-based models require an OpenAI API key. Set it as an environment variable:
        ```bash
        export OPENAI_API_KEY="your_key_here"
        ```

3.  **Train the CF Model**:
    *   If you haven't already, run the training script to generate the `svd_model.pkl` file. This only needs to be done once.
        ```bash
        python collaborative_filtering_bookcrossing.py
        ```

4.  **Run the Streamlit App**:
    *   Launch the interactive evaluation application:
        ```bash
        streamlit run app.py
        ```
    *   Open the provided URL in your browser, select a group recommendation strategy from the sidebar, and click "Run Evaluation Demo".

## 📂 Files in this Repository

- `app.py`: The main Streamlit application for evaluating the models.
- `collaborative_filtering_bookcrossing.py`: A script to train the SVD collaborative filtering model and save it.
- `README.md`: This file.
- `.gitignore`: Specifies files to be ignored by Git (e.g., `.venv`, `__pycache__`).

*(Note: The `BX-Book-Ratings.csv` and `BX-Books.csv` data files are required to run the project but are not included in this repository).*
