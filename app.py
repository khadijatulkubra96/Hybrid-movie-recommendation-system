import streamlit as st
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

# 1. Page Config & CSS Injection for Netflix-Style Interface
st.set_page_config(page_title="Movie Lounge", layout="wide", page_icon="🎬")

st.markdown("""
    <style>
    /* Main body background setting */
    .stApp {
        background: linear-gradient(135px, #0f2027, #203a43, #2c5364);
        color: #e0e0e0;
    }
    /* Control panel adjustments */
    section[data-testid="stSidebar"] {
        background-color: #111a24 !important;
        border-right: 2px solid #00d2ff;
    }
    /* High-end Neon Glow Glassmorphism Cards */
    .movie-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
        transition: transform 0.4s ease, border 0.4s ease;
    }
    .movie-card:hover {
        transform: translateY(-10px) scale(1.02);
        border: 1px solid #00d2ff;
        box-shadow: 0 0 20px rgba(0, 210, 255, 0.4);
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    .metric-box {
        background: rgba(0, 210, 255, 0.1);
        border-left: 4px solid #00d2ff;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🎬 Movie Lounge: Personalized Hybrid Recommendation Dashboard</h1>", unsafe_allow_html=True)

# Safe poster rendering engine
def fetch_poster_url(title):
    try:
        pure_title = title.split('(')[0].strip()
        # Direct backup secure access via TMDB public endpoint
        fallback_key = "8265bd1679663a7ea12ac168da84d2e8"
        query_url = f"https://api.themoviedb.org/3/search/movie?api_key={fallback_key}&query={pure_title}"
        api_response = requests.get(query_url, timeout=2).json()
        if api_response and 'results' in api_response and len(api_response['results']) > 0:
            path = api_response['results'][0].get('poster_path')
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"
    except:
        pass
    # High-res premium dark replacement vector graphic link if API encounters issues
    return "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=500"

# Dataset ingest validation
try:
    df_content = pd.read_csv('movies_cleaned.csv.csv')
    df_user = pd.read_csv('ratings_title.csv')
    df_user.rename(columns={'userId':'user_id', 'movieId':'movie_id'}, inplace=True)
    
    # Match layout semantics dynamically
    txt_col = 'tags' if 'tags' in df_content.columns else ('overview' if 'overview' in df_content.columns else 'fallback')
    if txt_col == 'fallback':
        df_content['fallback'] = df_content['genres'].fillna('') + ' ' + df_content['title'].fillna('')
        
    df_content[txt_col] = df_content[txt_col].fillna('')
    
    # Pre-calculate internal vectors seamlessly
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_mat = vectorizer.fit_transform(df_content[txt_col])
    similarity_features = cosine_similarity(tfidf_mat)
    df_content_sim = pd.DataFrame(similarity_features, index=df_content['title'].values, columns=df_content['title'].values)

except Exception as data_fault:
    st.error(f"Initialization Defect: {data_fault}")

# Interactive Control Layout on Left Panel
st.sidebar.markdown("<h2 style='color: #00d2ff; margin-top:0;'>🎛️ Engine Control</h2>", unsafe_allow_html=True)
sample_size = st.sidebar.number_input('Set fine-tuning footprint depth:', min_value=3, value=3, step=1)

selections = []
movie_pool = df_content['title'].values.tolist() if 'df_content' in locals() else []

st.sidebar.markdown("<p style='color: #a0aab2;'>Rate explicit context hooks below:</p>", unsafe_allow_html=True)
for row_idx in range(sample_size):
    selected_movie = st.sidebar.selectbox(f"Movie Anchor #{row_idx+1}", options=movie_pool, key=f"sel_{row_idx}")
    selected_rating = st.sidebar.select_slider("Assign Weight:", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], value=4.0, key=f"sld_{row_idx}")
    selections.append((selected_movie, selected_rating))

# Main Screen Real-Time Telemetry Cards
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"<div class='metric-box'><strong>System Mode:</strong> Hybrid (Content + Collaborative)</div>", unsafe_allow_html=True)
with col_m2:
    st.markdown(f"<div class='metric-box'><strong>Content Feature Extraction:</strong> TF-IDF Vectorizer</div>", unsafe_allow_html=True)
with col_m3:
    st.markdown(f"<div class='metric-box'><strong>Collaborative Distance Metric:</strong> Cosine Similarity</div>", unsafe_allow_html=True)

if st.sidebar.button('🚀 Compute Hybrid Recommendations'):
    with st.spinner('🎯 Matrix manipulation ongoing... parsing user vectors.'):
        try:
            # Inject custom vector points into transactional framework
            next_user_uid = df_user['user_id'].max() + 1
            batch_inputs = []
            for item, score in selections:
                batch_inputs.append({
                    'user_id': next_user_uid, 'rating': score,
                    'movie_id': df_content.loc[df_content['title'] == item, 'movie_id'].values[0] if 'movie_id' in df_content.columns else 0,
                    'title': item,
                    'genres': df_content.loc[df_content['title'] == item, 'genres'].values[0] if 'genres' in df_content.columns else 'General',
                    'year': df_content.loc[df_content['title'] == item, 'year'].values[0] if 'year' in df_content.columns else 2000
                })
            
            df_user_extended = pd.concat([df_user, pd.DataFrame(batch_inputs)]).drop_duplicates()
            
            # Pivot & Normalization phase
            u_i_matrix = df_user_extended.pivot_table(values='rating', index='user_id', columns='title')
            normalized_matrix = u_i_matrix.subtract(u_i_matrix.mean(axis=1), axis='rows')
            u_u_similarity = cosine_similarity(sparse.csr_matrix(normalized_matrix.fillna(0)))
            df_u_u_sim = pd.DataFrame(u_u_similarity, index=u_i_matrix.index, columns=u_i_matrix.index)

            # Mathematical execution functions
            def run_content_pipeline(uid):
                profile = df_user_extended[df_user_extended['user_id'] == uid]
                watched = profile['title'].values
                threshold = profile['rating'].mean()
                promising_anchors = [m for m in watched if profile[profile['title'] == m]['rating'].values[0] >= threshold]
                
                sub_lists = []
                for movie_anchor in promising_anchors:
                    if movie_anchor in df_content_sim.index:
                        scores = df_content_sim[movie_anchor].drop(watched, errors='ignore')
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
                    if den != 0:
                        computed_scores[target_col] = num / den
                return pd.DataFrame(computed_scores.items(), columns=['title', 'user_score'])

            c_outputs = run_content_pipeline(next_user_uid)
            u_outputs = run_collaborative_pipeline(next_user_uid, 0.05)
            
            if not c_outputs.empty and not u_outputs.empty:
                integrated = pd.merge(c_outputs, u_outputs, on='title')
                integrated['hybrid_rank'] = (integrated['content_score'] + integrated['user_score']) / 2
                top_hits = integrated.sort_values(by='hybrid_rank', ascending=False)[:10]
                final_payload = pd.merge(df_content, top_hits[['title', 'hybrid_rank']], on='title').sort_values(by='hybrid_rank', ascending=False)
                
                st.markdown("<h3 style='color: #00d2ff; margin-top:15px; margin-bottom:20px;'>🍿 Highly Confident Engine Matches For You:</h3>", unsafe_allow_html=True)
                
                # Double Row Visual Grid Allocation 
                card_index = 0
                for block_row in range(2):
                    grid_cols = st.columns(5)
                    for active_col in grid_cols:
                        if card_index < len(final_payload):
                            t = final_payload.iloc[card_index]['title']
                            g = final_payload.iloc[card_index]['genres'].split('|')[0] if 'genres' in final_payload.columns else 'Drama'
                            y = int(final_payload.iloc[card_index]['year']) if 'year' in final_payload.columns else 2024
                            img_src = fetch_poster_url(t)
                            
                            with active_col:
                                st.markdown(f"""
                                    <div class="movie-card">
                                        <img src="{img_src}" style="width:100%; object-fit: cover; border-radius:10px; margin-bottom:12px; height:240px;">
                                        <h5 style="color:#ffffff; font-size:14px; margin:4px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{t}</h5>
                                        <span style="color:#00d2ff; font-size:12px; font-weight:bold;">🎬 {g}</span>
                                        <span style="color:#a0aab2; font-size:12px;"> | 📅 {y}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            card_index += 1
            else:
                st.warning("Low spatial cross-referencing footprint found. Change target evaluation profiles to trigger neighborhood activation.")
        except Exception as system_fault:
            st.error(f"Runtime Processing Interrupt: {system_fault}")