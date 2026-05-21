import streamlit as st
import sys

# Force-import pandas and guarantee its alias globally
try:
    import pandas as pd
except ImportError:
    st.error("Pandas library is missing from the environment configuration.")

import pickle
import requests

# Try importing sklearn components safely
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    sklearn_available = True
except ImportError:
    sklearn_available = False

# --- 1. Premium Custom Layout & Theater Background ---
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
        font-size: 75px !important; 
        font-weight: 800; 
        text-align: center; 
        color: #E50914; 
        margin-top: -20px;
        text-shadow: 3px 3px 15px rgba(229, 9, 20, 0.4);
    }
    .sub-title { 
        font-size: 22px !important; 
        text-align: center; 
        color: #e0e0e0; 
        margin-bottom: 40px; 
    }
    h2, h3, h4, label, .stSelectbox p {
        color: white !important;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px;
        text-align: center;
        transition: transform 0.3s ease, border 0.3s ease;
    }
    .movie-card:hover {
        transform: scale(1.04);
        border: 1px solid #E50914;
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.4);
    }
    .stButton>button {
        background-color: #E50914 !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        width: 100%;
        padding: 10px 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Live API Functions ---
def get_live_data(type="trending"):
    if type == "trending":
        url = "https://api.themoviedb.org/3/trending/movie/day?api_key=8265bd1679663a7ea12ac168da84d2e8"
    else:
        url = "https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1"
        
    try:
        response = requests.get(url, timeout=3).json()
        return response.get('results', [])[:5] 
    except:
        return []

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url, timeout=2).json()
        if data and 'poster_path' in data and data['poster_path']:
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# --- 3. Secure Dataset Loading (With explicit namespace fallback) ---
@st.cache_resource
def load_data():
    # Local inline alias insurance to stop NameError inside cache workers
    import pandas as fallback_pd
    try:
        with open('movie_list.pkl', 'rb') as f:
            movies = pickle.load(f)
        
        if 'tags' not in movies.columns:
            movies['tags'] = movies['overview'].fillna('') + " " + movies['genres'].fillna('')
            
        if sklearn_available:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(movies['tags'].fillna(''))
            sim = cosine_similarity(tfidf_matrix)
            return movies, sim, True, ""
        else:
            return movies, None, False, "Scikit-learn dependencies not ready."
            
    except Exception as e:
        # Guarantee dataframe setup works even if file is entirely missing
        empty_df = fallback_pd.DataFrame(columns=['title', 'id', 'tags'])
        return empty_df, None, False, str(e)

movies, similarity, data_loaded, load_error = load_data()

# --- 4. Branding Header ---
st.markdown('<p class="main-title">🎬 Movie Lounge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Recommendations for Your Next Movie Night</p>', unsafe_allow_html=True)

# --- 5. SEARCH & RECOMMEND SECTION ---
if data_loaded and not movies.empty and similarity is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        selected_movie = st.selectbox("Search for a movie...", movies['title'].values)
        if st.button('Recommend'):
            movie_indices = movies[movies['title'] == selected_movie].index
            if len(movie_indices) > 0:
                idx = movies.index.get_loc(movie_indices[0])
                distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
                
                st.write("---")
                st.subheader(f"🍿 Recommendations for {selected_movie}")
                cols = st.columns(5)
                
                for i in range(1, 6):
                    if i < len(distances):
                        movie_idx = distances[i][0]
                        movie_id = movies.iloc[movie_idx].id
                        movie_title = movies.iloc[movie_idx].title
                        poster_url = fetch_poster(movie_id)
                        
                        with cols[i-1]:
                            st.markdown(f"""
                                <div class="movie-card">
                                    <img src="{poster_url}" style="width:100%; height:240px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                                    <p style="font-size:13px; font-weight:bold; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{movie_title}</p>
                                </div>
                                """, unsafe_allow_html=True)
else:
    # Beautiful error logging that doesn't break the live UI dashboard
    st.warning("📊 Local AI Search Engine is asleep because 'movie_list.pkl' wasn't read. Enjoy our Live Global sections below!")
    if load_error:
        st.info(f"Technical Log: {load_error}")

st.write("---")

# --- 6. LIVE DYNAMIC SECTIONS (TMDB API Based) ---

# Live Trending Section
st.subheader("🔥 Trending Globally Today")
live_trending = get_live_data("trending")
if live_trending:
    t_cols = st.columns(5)
    for i, m in enumerate(live_trending):
        poster_path = m.get('poster_path')
        img_url = f"https://image.tmdb.org/t/p/w500/{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
        with t_cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{img_url}" style="width:100%; height:240px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                    <p style="font-size:13px; font-weight:bold; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{m.get('title', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

st.write("---")

# Live Top Rated Section
st.subheader("⭐ All-Time Classics (Live)")
live_top = get_live_data("top_rated")
if live_top:
    r_cols = st.columns(5)
    for i, m in enumerate(live_top):
        poster_path = m.get('poster_path')
        img_url = f"https://image.tmdb.org/t/p/w500/{p_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
        with r_cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{img_url}" style="width:100%; height:240px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                    <p style="font-size:13px; font-weight:bold; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{m.get('title', 'Unknown')} ({m.get('vote_average', 0)})</p>
                </div>
                """, unsafe_allow_html=True)