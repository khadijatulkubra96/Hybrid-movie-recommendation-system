import streamlit as st
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Movie Lounge", layout="wide")
st.header('🎬 Personalized Movie Recommendations')

# Data imports
try:
    df_content = pd.read_csv('movies_cleaned.csv.csv')
    df_user = pd.read_csv('ratings_title.csv')
    df_user.rename(columns={'userId':'user_id', 'movieId':'movie_id'}, inplace=True)
    
    # DYNAMIC CALCULATION: Instead of loading a missing .pkl file, we compute it live!
    st.info("Initializing recommendation matrix from dataset...")
    
    # 1. Check if 'tags' or 'genres' column exists to build TF-IDF
    if 'tags' in df_content.columns:
        text_column = 'tags'
    elif 'overview' in df_content.columns:
        text_column = 'overview'
    else:
        # Fallback if specific text features aren't combined yet
        df_content['fallback_tags'] = df_content['genres'].fillna('') + ' ' + df_content['title'].fillna('')
        text_column = 'fallback_tags'
        
    df_content[text_column] = df_content[text_column].fillna('')
    
    # 2. Compute TF-IDF Matrix on the fly
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_content[text_column])
    
    # 3. Generate Cosine Similarity DataFrame dynamically
    content_similarity = cosine_similarity(tfidf_matrix)
    df_content_sim = pd.DataFrame(content_similarity, index=df_content['title'].values, columns=df_content['title'].values)
    st.success("Engine ready!")

except Exception as e:
    st.error(f"Data loading error: {e}. Please ensure CSV files are properly uploaded to GitHub.")

# Get data from the user
new_user_data = []
number = st.sidebar.number_input('How many movies would you like to rate?', min_value=3, value=3, step=1)

current_line_number = 0
options = df_content['title'].values.tolist() if 'df_content' in locals() else []

st.sidebar.subheader("Rate Movies to Train the Hybrid Engine:")
for _ in range(number):
    movie = st.sidebar.selectbox('Movie title', key=str(current_line_number), options=options)
    rating = st.sidebar.select_slider('Rate the movie', options=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], key=str(current_line_number) + "_slider")
    new_user_data.append((movie, rating))
    current_line_number += 1

if st.sidebar.button('Get Recommendations'):
    with st.spinner('Calculating hybrid recommendations...'):
        # Add new_user_data to user database
        new_userId = df_user['user_id'].sort_values().values[-1] + 1
        new_user = []
        for movie, rating in new_user_data:
            new_ratings = {}
            new_ratings['user_id'] = new_userId
            new_ratings['rating'] = rating
            new_ratings['movie_id'] = df_content.loc[df_content['title'] == movie, 'movie_id'].values[0]
            new_ratings['title'] = movie
            new_ratings['genres'] = df_content.loc[df_content['title'] == movie, 'genres'].values[0]
            new_ratings['year'] = df_content[df_content['title'] == movie]['year'].values[0]
            new_user.append(new_ratings)

        df_new_user = pd.DataFrame(new_user).drop_duplicates()
        df_user = pd.concat([df_user, df_new_user])

        # Create User-Item Matrix
        user_item = df_user.pivot_table(values='rating', index='user_id', columns='title')

        # Normalize User-Item matrix
        norm_user_item = user_item.subtract(user_item.mean(axis=1), axis='rows')

        # User-User similarity matrix
        user_similarity = cosine_similarity(sparse.csr_matrix(norm_user_item.fillna(0)))
        df_user_sim = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)

        def get_content_similar_movies(user):
            df_current_user = df_user[df_user['user_id'] == user]
            user_watched_movies = df_current_user['title'].values
            user_mean_rating = df_current_user['rating'].mean()
            
            user_movies = []
            for movie in user_watched_movies:
                if df_current_user[df_current_user['title'] == movie]['rating'].values[0] >= user_mean_rating:
                    user_movies.append(movie)
                    
            similar_movies_list = []
            for movie in user_movies:
                if movie in df_content_sim.index:
                    series_sim = df_content_sim[movie].drop(user_watched_movies, errors='ignore')
                    similar_movies_list.append(series_sim.to_frame().T)
            
            if similar_movies_list:
                similar_movies = pd.concat(similar_movies_list, axis=0)
                content_rec = pd.DataFrame(similar_movies.sum()).reset_index().rename(columns={'index': 'title', 0: 'content_similarity'})
            else:
                content_rec = pd.DataFrame(columns=['title', 'content_similarity'])
                
            return pd.merge(df_content[['title', 'genres']], content_rec, how='inner').sort_values(by='content_similarity', ascending=False)

        def get_user_similar_movies(user, similarity_threshold):
            similar_users = df_user_sim[df_user_sim[user] > similarity_threshold][user].sort_values(ascending=False)[1:]
            target_user_movies = norm_user_item[norm_user_item.index == user].dropna(axis=1, how='all')
            similar_user_movies = norm_user_item[norm_user_item.index.isin(similar_users.index)].dropna(axis=1, how='all')
            
            for column in target_user_movies.columns: 
                if column in similar_user_movies.columns:
                    similar_user_movies.drop(column, axis=1, inplace=True)
                    
            movie_score = {}
            for movie in similar_user_movies.columns:
                movie_rating = similar_user_movies[movie]
                numerator = 0
                denominator = 0
                for u in similar_users.index:
                    if pd.notnull(movie_rating[u]):
                        weighted_score = similar_users[u] * movie_rating[u]
                        numerator += weighted_score
                        denominator += similar_users[u]
                if denominator != 0:
                    movie_score[movie] = numerator / denominator
                    
            movie_score_df = pd.DataFrame(movie_score.items(), columns=['title', 'user_similarity'])
            user_rec = pd.merge(df_content[['title', 'genres', 'year']], movie_score_df[['title', 'user_similarity']], how='inner')
            return user_rec.sort_values(by=['user_similarity', 'year'], ascending=False)

        def hybrid_recommender(user):
            content_df = get_content_similar_movies(user)
            collaborative_df = get_user_similar_movies(user, 0.1)
            
            content_user_scores = pd.merge(content_df, collaborative_df, on=['title', 'genres'])
            content_user_scores['similarity_score'] = (content_user_scores['content_similarity'] + content_user_scores['user_similarity']) / 2
            top_scores = content_user_scores.sort_values(by='similarity_score', ascending=False)[:10]
            
            available_cols = ['title', 'genres', 'year']
            for col in ['imdb_rating', 'tmdb_rating', 'vote_average']:
                if col in df_content.columns:
                    available_cols.append(col)
                    
            recommendations = pd.merge(df_content[available_cols], top_scores[['title', 'similarity_score']], on='title')
            return recommendations.sort_values(by='similarity_score', ascending=False)

        results = hybrid_recommender(new_userId)
        
        if not results.empty:
            st.subheader("Top 10 Recommendations For You:")
            st.table(results)
        else:
            st.warning("Not enough overlap data to compute hybrid scores. Try rating different popular movies!")