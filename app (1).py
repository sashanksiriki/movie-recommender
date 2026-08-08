import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

# ------------------------------------------------------------------
# One-time NLTK downloads (cached so it only runs once per session)
# ------------------------------------------------------------------
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


stop_words = download_nltk_data()


def preprocess_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)


# ------------------------------------------------------------------
# Load + preprocess data + build similarity matrix (cached)
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing movie data...")
def load_data():
    df = pd.read_csv("movies.csv")

    required_columns = ["genres", "keywords", "overview", "title"]
    df = df[required_columns].dropna().reset_index(drop=True)

    df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]
    data = df[["title", "combined"]].copy()
    data["cleaned_text"] = data["combined"].apply(preprocess_text)

    return data


@st.cache_resource(show_spinner="Building similarity model...")
def build_model(data: pd.DataFrame):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["cleaned_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def recommend_movies(movie_name: str, data: pd.DataFrame, cosine_sim, top_n: int = 5):
    idx = data[data["title"].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return None
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1] * 100, 1) for i in sim_scores]

    result = data[["title"]].iloc[movie_indices].copy()
    result["match %"] = scores
    return result.reset_index(drop=True)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🎬 Movie Recommender")
st.write("Get similar movie recommendations based on genre, keywords, and plot overview.")

data = load_data()
cosine_sim = build_model(data)

movie_list = sorted(data["title"].unique().tolist())

movie_name = st.selectbox(
    "Pick a movie you like:",
    options=movie_list,
    index=None,
    placeholder="Start typing a movie title...",
)

top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=5)

if st.button("🚀 Recommend Similar Movies", type="primary"):
    if not movie_name:
        st.warning("Please select a movie first.")
    else:
        recommendations = recommend_movies(movie_name, data, cosine_sim, top_n=top_n)
        if recommendations is None:
            st.error("Movie not found in the dataset.")
        else:
            st.subheader(f"Because you liked *{movie_name}*:")
            for i, row in recommendations.iterrows():
                st.write(f"**{i + 1}. {row['title']}**  —  {row['match %']}% match")

st.divider()
st.caption("Built with Streamlit · TF-IDF + Cosine Similarity")
