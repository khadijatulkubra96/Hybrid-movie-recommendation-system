import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. GLOBAL VARIABLES INITIALIZATION (Prevents NameError completely) ---
df_content = pd.DataFrame(columns=['title', 'movie_id', 'genres', 'year'])
df_user = pd.DataFrame(columns=['user_id', 'rating', 'movie_id', 'title', 'genres', 'year'])
df_content_sim = pd.DataFrame()
movie_pool = []
init_success = False
error_msg = ""

# --- 2. Premium Custom Layout & Theater Background ---
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
    h2, h3, h4, .stSlider, label {
        color: white !important;
    }
    div[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 1px solid #E50914;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(8px);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
        text-align: center;
        transition: transform 0.3s ease, border 0.3s ease;
    }
    .movie-card:hover {
        transform: scale(1.03);
        border: 1px solid #E50914;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Live API Functions ---
def get_live_data(endpoint_type="trending"):
    if endpoint_type == "trending":
        url = "https://api.themoviedb.org/3/trending/movie/day?api_key=8265bd1679663a7ea12ac168da84d2e8"
    else:
        url = "https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1"
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
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except:
        pass
    return "https://via.placeholder.com/500x750?text=No+Poster"

# --- 4. Safe Local Data Loading Block ---
try:
    df_content = pd.read_csv('movies_cleaned.csv.csv')
    df_user = pd.read_csv('ratings_title.csv')
    df_user.rename(columns={'userId':'user_id', 'movieId':'movie_id'}, inplace=True)
    
    txt_col = 'tags' if 'tags' in df_content.columns else ('overview' if 'overview' in df_content.columns else 'fallback')
    if txt_col == 'fallback':
        df_content['fallback'] = df_content['genres'].fillna('') + ' ' + df_content['title'].fillna('')
        
    df_content[txt_col] = df_content[txt_col].fillna('')
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_mat = vectorizer.fit_transform(df_content[txt_col])
    similarity_features = cosine_similarity(tfidf_mat)
    df_content_sim = pd.DataFrame(similarity_features, index=df_content['title'].values, columns=df_content['title'].values)
    
    movie_pool = df_content['title'].values.tolist()
    init_success = True
except Exception as e:
    error_msg = str(e)

# --- 5. Render App UI Layout ---
st.markdown('<p class="main-title">🎬 Movie Lounge</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Recommendations for Your Next Movie Night</p>', unsafe_allow_html=True)

# Show error on main dashboard visually if data files failed to load
if not init_success:
    st.error(f"⚠️ Data Loading Alert: {error_msg}")
    st.info("System is currently running on Live Global API mode because local CSV tracking failed.")

# --- 6. Sidebar Tuning Controls ---
st.sidebar.markdown("<h3 style='color: #E50914;'>🎛️ Hybrid Tuning</h3>", unsafe_allow_html=True)
sample_size = st.sidebar.number_input('Rate items to build anchor profile:', min_value=3, value=3, step=1)

selections = []
for row_idx in range(sample_size):
    if init_success and movie_pool:
        selected_movie = st.sidebar.selectbox(f"Movie Slot #{row_idx+1}", options=movie_pool, key=f"sel_{row_idx}")
        selected_rating = st.sidebar.select_slider("Rating Weight:", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=4.0, key=f"sld_{row_idx}")
        selections.append((selected_movie, selected_rating))
    else:
        st.sidebar.warning(f"Slot #{row_idx+1} locked (Waiting for CSV data)")

compute_clicked = st.sidebar.button('🚀 Compute Hybrid Model') if init_success else False

# --- SECTION A: LIVE ENGINE PREDICTIONS ---
if compute_clicked and init_success and not df_content_sim.empty:
    with st.spinner('🎯 Constructing user vectors and cross-referencing neighborhoods...'):
        try:
            next_user_uid = df_user['user_id'].max() + 1 if not df_user.empty else 1
            batch_inputs = []
            for item, score in selections:
                m_id_vals = df_content.loc[df_content['title'] == item, 'movie_id'].values
                gen_vals = df_content.loc[df_content['title'] == item, 'genres'].values
                yr_vals = df_content.loc[df_content['title'] == item, 'year'].values
                
                batch_inputs.append({
                    'user_id': next_user_uid, 'rating': score,
                    'movie_id': m_id_vals[0] if len(m_id_vals) > 0 else 0,
                    'title': item,
                    'genres': gen_vals[0] if len(gen_vals) > 0 else 'General',
                    'year': yr_vals[0] if len(yr_vals) > 0 else 2024
                })
            
            df_user_extended = pd.concat([df_user, pd.DataFrame(batch_inputs)]).drop_duplicates()
            u_i_matrix = df_user_extended.pivot_table(values='rating', index='user_id', columns='title')
            normalized_matrix = u_i_matrix.subtract(u_i_matrix.mean(axis=1), axis='rows')
            u_u_similarity = cosine_similarity(sparse.csr_matrix(normalized_matrix.fillna(0)))
            df_u_u_sim = pd.DataFrame(u_u_similarity, index=u_i_matrix.index, columns=u_i_matrix.index)

            def run_content_pipeline(uid):
                profile = df_user_extended[df_user_extended['user_id'] == uid]
                watched = profile['title'].values
                threshold = profile['rating'].mean()
                promising_anchors = [m for m in watched if profile[profile['title'] == m]['rating'].values[0] >= threshold]
                sub_lists = []
                for m in promising_anchors:
                    if m in df_content_sim.index:
                        scores = df_content_sim[m].drop(watched, errors='ignore')
                        sub_lists.append(scores.to_frame().T)
                if sub_lists:
                    return pd.DataFrame(pd.concat(sub_lists).sum()).reset_index().rename(columns={'index': 'title', 0: 'content_score'})
                return pd.DataFrame(columns=['title', 'content_score'])

            def run_collaborative_pipeline(uid, limit=0.1):
                neighbors = df_u_u_sim[df_u_u_sim[uid] > limit][uid].sort_values(ascending=False)[1:]
                sub_matrix = normalized_matrix[normalized_matrix.index.isin(neighbors.index)].dropna(axis=1, how='all')
                my_movies = normalized_matrix[normalized_matrix.index == uid].dropna(axis=1, how='all').columns
                sub_matrix.drop(columns=my_movies, errors='ignore', inplace=True)
                computed_scores = {}
                for target_col in sub_matrix.columns:
                    series_ratings = sub_matrix[target_col]
                    num, den = 0, 0
                    for peer in neighbors.index:
                        if pd.notnull(series_ratings[peer]):
                            num += neighbors[peer] * series_ratings[peer]
                            den += neighbors[peer]
                    if den != 0: computed_scores[target_col] = num / den
                return pd.DataFrame(computed_scores.items(), columns=['title', 'user_score'])

            c_outputs = run_content_pipeline(next_user_uid)
            u_outputs = run_collaborative_pipeline(next_user_uid, 0.05)
            
            if not c_outputs.empty and not u_outputs.empty:
                integrated = pd.merge(c_outputs, u_outputs, on='title')
                integrated['hybrid_rank'] = (integrated['content_score'] + integrated['user_score']) / 2
                top_hits = integrated.sort_values(by='hybrid_rank', ascending=False)[:5]
                final_payload = pd.merge(df_content, top_hits[['title', 'hybrid_rank']], on='title').sort_values(by='hybrid_rank', ascending=False)
                
                st.subheader("🎯 Handpicked For You (AI Engine Predictions)")
                h_cols = st.columns(5)
                for idx, col in enumerate(h_cols):
                    if idx < len(final_payload):
                        t = final_payload.iloc[idx]['title']
                        img_src = fetch_poster_by_title(t)
                        with col:
                            st.markdown(f"""
                                <div class="movie-card">
                                    <img src="{img_src}" style="width:100%; height:260px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                                    <p style="font-size:13px; font-weight:bold; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{t}</p>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.warning("Neighborhood densities sparse. Tweak rating scopes on sidebar.")
        except Exception as ex:
            st.error(f"Pipeline Interruption: {ex}")

st.write("---")

# --- SECTION B: LIVE GLOBAL TRENDING (TMDB API) ---
st.subheader("🔥 Trending Globally Today")
live_trending = get_live_data("trending")
if live_trending:
    t_cols = st.columns(5)
    for i, m in enumerate(live_trending):
        with t_cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="https://image.tmdb.org/t/p/w500/{m.get('poster_path', '')}" style="width:100%; height:260px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                    <p style="font-size:13px; font-weight:bold; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{m.get('title', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

st.write("---")

# --- SECTION C: ALL-TIME CLASSICS ---
st.subheader("⭐ All-Time Classics (Live)")
live_top = get_live_data("top_rated")
if live_top:
    r_cols = st.columns(5)
    for i, m in enumerate(live_top):
        with r_cols[i]:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="https://image.tmdb.org/t/p/w500/{m.get('poster_path', '')}" style="width:100%; height:260px; object-fit:cover; border-radius:6px; margin-bottom:8px;">
                    <p style="font-size:13px; font-weight:bold; margin:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{m.get('title', 'Unknown')} ({m.get('vote_average', 0)})</p>
                </div>
                """, unsafe_allow_html=True)