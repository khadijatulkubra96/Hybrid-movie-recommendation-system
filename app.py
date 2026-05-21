import streamlit as st
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

# 1. Custom Page Configuration & Premium Theme (CSS Styling)
st.set_page_config(page_title="Movie Lounge", layout="wide", page_icon="🎬")

st.markdown("""
    <style>
    .main {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    .stButton>button {
        background-color: #45f3ff !important;
        color: #0b0c10 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        box-shadow: 0 0 15px #45f3ff;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px #45f3ff;
    }
    h1 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        text-shadow: 2px 2px 8px #45f3ff;
    }
    .movie-card {
        background-color: #1f2833;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #45f3ff;
        text-align: center;
        margin-bottom: 20px;
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🎬 Movie Lounge: Personalized Hybrid Engine')

# Helper function to fetch posters safely from TMDb API
def get_movie_poster(movie_title):
    try:
        # Cleaning year format from title if present (e.g. "Toy Story (1995)" -> "Toy Story")
        clean_title = movie_title.split('(')[0].strip()
        api_key = "8265bd1679663a7ea12ac168da84d2e8" # Public TMDB demo fallback key
        url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
        response = requests.get(url, timeout=5).json()
        if response['results']:
            poster_path = response['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=500" # High quality placeholder

# Data imports
try:
    df_content = pd.read_csv('movies_cleaned.csv.csv')
    df_user = pd.read_csv('ratings_title.csv')
    df_user.rename(columns={'userId':'user_id', 'movieId':'movie_id'}, inplace=True)
    
    # Check text features
    if 'tags' in df_content.columns:
        text_column = 'tags'
    elif 'overview' in df_content.columns:
        text_column = 'overview'
    else:
        df_content['fallback_tags'] = df_content['genres'].fillna('') + ' ' + df_content['title'].fillna('')
        text_column = 'fallback_tags'
        
    df_content[text_column] = df_content[text_column].fillna('')
    
    # Fit TF-IDF Matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_content[text_column])
    content_similarity = cosine_similarity(tfidf_matrix)
    df_content_sim = pd.DataFrame(content_similarity, index=df_content['title'].values, columns=df_content['title'].values)

except Exception as e:
    st.error(f"Data loading error: {e}")

# Sidebar User Interface
st.sidebar.markdown("<h2 style='color: #45f3ff;'>🎛️ Control Panel</h2>", unsafe_allow_html=True)
number = st.sidebar.number_input('How many movies would you like to rate?', min_value=3, value=3, step=1)

new_user_data = []
options = df_content['title'].values.tolist() if 'df_content' in locals() else []

st.sidebar.subheader("Rate Movies:")
current_line_number = 0
for _ in range(number):
    movie = st.sidebar.selectbox('Movie title', key=str(current_line_number), options=options)
    rating = st.sidebar.select_slider('Rate', options=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], key=str(current_line_number) + "_slider")
    new_user_data.append((movie, rating))
    current_line_number += 1

if st.sidebar.button('Get Hybrid Recommendations'):
    with st.spinner('🎬 Exploring user neighborhood and matching descriptions...'):
        new_userId = df_user['user_id'].sort_values().values[-1] + 1
        new_user = []
        for movie, rating in new_user_data:
            new_ratings = {
                'user_id': new_userId,
                'rating': rating,
                'movie_id': df_content.loc[df_content['title'] == movie, 'movie_id'].values[0],
                'title': movie,
                'genres': df_content.loc[df_content['title'] == movie, 'genres'].values[0],
                'year': df_content[df_content['title'] == movie]['year'].values[0]
            }
            new_user.append(new_ratings)

        df_user = pd.concat([df_user, pd.DataFrame(new_user).drop_duplicates()])
        user_item = df_user.pivot_table(values='rating', index='user_id', columns='title')
        norm_user_item = user_item.subtract(user_item.mean(axis=1), axis='rows')
        user_similarity = cosine_similarity(sparse.csr_matrix(norm_user_item.fillna(0)))
        df_user_sim = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)

        def get_content_similar_movies(user):
            df_current_user = df_user[df_user['user_id'] == user]
            user_watched_movies = df_current_user['title'].values
            user_mean_rating = df_current_user['rating'].mean()
            
            user_movies = [m for m in user_watched_movies if df_current_user[df_current_user['title'] == m]['rating'].values[0] >= user_mean_rating]
            
            similar_movies_list = []
            for m in user_movies:
                if m in df_content_sim.index:
                    series_sim = df_content_sim[m].drop(user_watched_movies, errors='ignore')
                    similar_movies_list.append(series_sim.to_frame().T)
            
            if similar_movies_list:
                similar_movies = pd.concat(similar_movies_list, axis=0)
                return pd.DataFrame(similar_movies.sum()).reset_index().rename(columns={'index': 'title', 0: 'content_similarity'})
            return pd.DataFrame(columns=['title', 'content_similarity'])

        def get_user_similar_movies(user, similarity_threshold):
            similar_users = df_user_sim[df_user_sim[user] > similarity_threshold][user].sort_values(ascending=False)[1:]
            similar_user_movies = norm_user_item[norm_user_item.index.isin(similar_users.index)].dropna(axis=1, how='all')
            
            target_movies = norm_user_item[norm_user_item.index == user].dropna(axis=1, how='all').columns
            similar_user_movies.drop(columns=target_movies, errors='ignore', inplace=True)
                    
            movie_score = {}
            for m in similar_user_movies.columns:
                movie_rating = similar_user_movies[m]
                num, den = 0, 0
                for u in similar_users.index:
                    if pd.notnull(movie_rating[u]):
                        num += similar_users[u] * movie_rating[u]
                        den += similar_users[u]
                if den != 0:
                    movie_score[m] = num / den
                    
            return pd.DataFrame(movie_score.items(), columns=['title', 'user_similarity'])

        def hybrid_recommender(user):
            c_df = get_content_similar_movies(user)
            u_df = get_user_similar_movies(user, 0.1)
            if c_df.empty or u_df.empty: return pd.DataFrame()
            
            content_user_scores = pd.merge(c_df, u_df, on='title')
            content_user_scores['similarity_score'] = (content_user_scores['content_similarity'] + content_user_scores['user_similarity']) / 2
            top_scores = content_user_scores.sort_values(by='similarity_score', ascending=False)[:10]
            
            return pd.merge(df_content[['title', 'genres', 'year']], top_scores[['title', 'similarity_score']], on='title')

        results = hybrid_recommender(new_userId)
        
        if not results.empty:
            st.markdown("<h3 style='color: #45f3ff; margin-top:20px;'>🍿 Your Custom Recommendations</h3>", unsafe_allow_html=True)
            
            # Creating Grid Columns for Posters (5 columns layout)
            idx = 0
            for row in range(2):
                cols = st.columns(5)
                for col in cols:
                    if idx < len(results):
                        movie_title = results.iloc[idx]['title']
                        genres = results.iloc[idx]['genres'].split('|')[0]
                        year = results.iloc[idx]['year']
                        poster_url = get_movie_poster(movie_title)
                        
                        with col:
                            st.markdown(f"""
                                <div class="movie-card">
                                    <img src="{poster_url}" style="width:100%; border-radius:8px; margin-bottom:10px;">
                                    <h4 style="color:#ffffff; font-size:14px; margin:5px 0;">{movie_title}</h4>
                                    <p style="color:#45f3ff; font-size:12px; margin:0;">🎬 {genres} | 📅 {int(year)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        idx += 1
        else:
            st.warning("Not enough user overlap found. Try rating highly mainstream blockbusters!")