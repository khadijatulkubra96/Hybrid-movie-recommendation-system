import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Page Configuration
st.set_page_config(page_title="Movie Lounge", layout="wide")

# TMDB API Key (Poster images fetch karne ke liye)
API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    if 'poster_path' in data and data['poster_path']:
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    return "https://via.placeholder.com/500x750?text=No+Poster"

# Load Data (Jo files aapne upload ki hain unka data use hoga)
@st.cache_data
def load_data():
    # Note: File paths ko apne folder structure ke mutabiq adjust karein
    df = pd.read_csv('movies_cleaned.csv') 
    return df

try:
    movies = load_data()

    # Title
    st.title("🎬 Movie Lounge: Hybrid Recommender")
    st.markdown("---")

    # Sidebar for Selection
    selected_movie = st.sidebar.selectbox(
        "Apni pasand ki movie select karein:",
        movies['title'].values
    )

    # Logic: Recommendation Engine
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend(movie_title):
        idx = movies[movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
        
        rec_movies = []
        rec_posters = []
        
        for i in sim_scores:
            movie_id = movies.iloc[i[0]].tmdb_id
            rec_movies.append(movies.iloc[i[0]].title)
            rec_posters.append(fetch_poster(movie_id))
        
        return rec_movies, rec_posters

    if st.sidebar.button("Recommend"):
        names, posters = recommend(selected_movie)
        
        st.subheader(f"Movies similar to: {selected_movie}")
        
        # Displaying in columns
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.image(posters[i])
                st.text(names[i])
        
        cols2 = st.columns(5)
        for i in range(5, 10):
            with cols2[i-5]:
                st.image(posters[i])
                st.text(names[i])

except Exception as e:
    st.error(f"Error: {e}. Please ensure 'movies_cleaned.csv' is uploaded.")