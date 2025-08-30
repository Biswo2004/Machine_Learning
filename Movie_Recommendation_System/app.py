# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Config
# ------------------------------
CSV_FILE = "merged_dataset.csv"  # Path to your CSV
TOP_N = 5  # Number of recommendations

# ------------------------------
# Load dataset
# ------------------------------
@st.cache_data
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    df.fillna("", inplace=True)
    return df

df = load_data(CSV_FILE)

# ------------------------------
# Build TF-IDF & Cosine Similarity
# ------------------------------
@st.cache_resource
def build_similarity_model(df):
    # Combine features
    df["combined_features"] = df["name"] + " " + df["genres"] + " " + df["year"].astype(str)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_similarity_model(df)

# ------------------------------
# Recommendation function
# ------------------------------
def get_recommendations(title, top_n=TOP_N):
    title_lower = title.lower()
    df_titles_lower = df["name"].str.lower()

    if title_lower not in df_titles_lower.values:
        return []  # Movie not found

    idx = df_titles_lower[df_titles_lower == title_lower].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the movie itself

    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][["name", "year", "genres", "rating"]]

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Type a movie name and get recommendations based on genres, year, and rating.")

# Search input
movie_title = st.text_input("Enter a movie name", "")

if st.button("Get Recommendations"):
    if not movie_title.strip():
        st.warning("Please enter a movie name.")
    else:
        recommended_df = get_recommendations(movie_title)
        if recommended_df.empty:
            st.error(f"Movie '{movie_title}' not found in dataset.")
        else:
            st.subheader(f"Movies similar to **{movie_title}**:")
            cols = st.columns(3)
            for i, (_, row) in enumerate(recommended_df.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"**{row['name']} ({row['year']})**")
                    st.caption(f"Genres: {row['genres']}\nRating: {row.get('rating', 'N/A')} ⭐")
