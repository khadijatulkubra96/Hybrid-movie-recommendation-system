import streamlit as st
import numpy as np
import pandas as pd
import requests
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. STATE LOCKING ---
if 'rec_results' not in st.session_state:
    st.session_state.rec_results = []

# --- 2. Premium Layout Styling ---
st.set_page_config(page_title="Movie Lounge", layout="wide", page_icon="🎬")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-attachment: fixed;
    }
    .main-title { 
        font-size: 55px !important; 
        font-weight: 800; 
        text-align: center; 
        color: #E50914; 
        margin-top: -10px;
    }
    .sub-title { 
        font-size: 18px !important; 
        text-align: center; 
        color: #e0e0e0; 
        margin-bottom: 30px; 
    }
    h2, h3, h4, label, p, .stSlider { color: white !important; }
    div[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid #E50914;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px;
        text-align: center;
    }
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Live TMDB API Fallbacks ---
def get_live_data(endpoint_type="trending"):
    url = "https://api.themoviedb.org/3/trending/movie/day?api_key=8265bd1679663a7ea12ac168da84d2e8" if endpoint_type == "trending" else "https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1"
    try:
        response = requests.get(url, timeout=3).json()
        return response.get('results', [])[:5]
    except:
        return []

def fetch_poster_by_title(movie_title):
    try:
        clean_title = movie_title.split('(')[0].strip()
        url = f"https://api.themoviedb.org/3/search/movie?api_key=8265bd1679663a7ea12ac168da84d2e8&query={clean_title}"
        response = requests.get(url, timeout=2).json()
        if response.get('results'):
            path = response['results'][0].get('poster_path')
            if path: return f"https://image.tmdb.org/t/p/w500{path}"
    except: pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# --- 4. DATA ENGINE LOADING ---
df_content = pd.DataFrame()
df_content_sim = pd.DataFrame()
movie_pool = []
init_success = False

for f in os.listdir('.'):
    if 'movies_cleaned' in f or 'clean_content' in f:
        try:
            df_content = pd.read_csv(f)
            txt_col = 'genres' if 'genres' in df_content.columns else df_content.columns[1]
            df_content[txt_col] = df_content[txt_col].fillna('')
            
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_mat = vectorizer.fit_transform(df_content[txt_col])
            similarity_features = cosine_similarity(tfidf_mat)
            df_content_sim = pd.DataFrame(similarity_features, index=df_content['title'].values, columns=df_content['title'].values)
            
            movie_pool = sorted(df_content['title'].dropna().unique().tolist())
            init_success = True
            break
        except:
            pass

# --- 5. Render App UI ---
st.markdown('<p class="main-title">🎬 Movie Lounge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Recommendations for Your Next Movie Night</p>', unsafe_allow_html=True)

# --- 6. Sidebar Settings ---
st.sidebar.markdown("<h3 style='color: #E50914;'>🎛️ Hybrid Tuning</h3>", unsafe_allow_html=True)
sample_size = st.sidebar.number_input('How many movies to rate?', min_value=1, max_value=5, value=3, step=1)

selections = []
if init_success and movie_pool:
    st.sidebar.subheader("Rate Movies:")
    for row_idx in range(sample_size):
        d_idx = min(row_idx * 25, len(movie_pool) - 1)
        selected_movie = st.sidebar.selectbox(f"Movie #{row_idx+1}", options=movie_pool, index=d_idx, key=f"s_{row_idx}")
        selected_rating = st.sidebar.slider(f"Rating #{row_idx+1}:", 0.5, 5.0, 4.0, step=0.5, key=f"v_{row_idx}")
        selections.append((selected_movie, selected_rating))
    
    if st.sidebar.button('Get Recommendations'):
        watched_movies = [m for m, r in selections]
        score_acc = pd.Series(0.0, index=df_content_sim.index)
        for m, rating in selections:
            if m in df_content_sim.index:
                score_acc = score_acc.add(df_content_sim[m] * (rating - 3.0), fill_value=0)
        
        score_acc.drop(index=watched_movies, errors='ignore', inplace=True)
        st.session_state.rec_results = list(score_acc.sort_values(ascending=False).head(5).index)

else:
    st.sidebar.warning("Waiting for clean_content.csv parsing...")

# --- 7. DISPLAY AI RECOMMENDATIONS ---
if st.session_state.rec_results:
    st.write("---")
    st.subheader("🎯 Top Recommendations Tailored For You:")
    h_cols = st.columns(5)
    for idx, title in enumerate(st.session_state.rec_results):
        img_src = fetch_poster_by_title(title)
        with h_cols[idx]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{img_src}" style="width:100%; height:240px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                    <p style="font-size:12px; font-weight:bold; margin:0; overflow:hidden; text-overflow:ellipsis;">{title}</p>
                </div>
                """, unsafe_allow_html=True)

st.write("---")

# --- 8. LIVE SECTIONS ---
st.subheader("🔥 Trending Globally Today")
live_trending = get_live_data("trending")
if live_trending:
    t_cols = st.columns(5)
    for i, m in enumerate(live_trending):
        p_path = m.get('poster_path')
        img_url = f"https://image.tmdb.org/t/p/w500/{p_path}" if p_path else "https://via.placeholder.com/500x750?text=No+Poster"
        with t_cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{img_url}" style="width:100%; height:240px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                    <p style="font-size:12px; font-weight:bold; margin:0;">{m.get('title', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

# 🚨 BILKUL SAAF CODE - NO MANUAL SERVER COMMANDS AT THE BOTTOM