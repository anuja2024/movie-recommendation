import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from recommendation_engine import HybridRecommender
from data_pipeline_latest import DataPipelineLatest


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="film",
    layout="wide"
)

@st.cache_resource
def load_recommender():
    data_dir = Path('./data')
    
    if not (data_dir / 'recent_movies_ratings.csv').exists():
        st.warning("Training model... This may take a few minutes.")
        pipeline = DataPipelineLatest(data_dir=str(data_dir))
        ratings_df, movies_df = pipeline.load_movielens_latest()
        ratings_df, movies_df = pipeline.filter_recent_movies(ratings_df, movies_df)
        ratings_df, movies_df = pipeline.preprocess_data(ratings_df, movies_df)
        pipeline.save_data(ratings_df, movies_df)
    
    pipeline = DataPipelineLatest(data_dir=str(data_dir))
    ratings_df, movies_df = pipeline.load_processed_data()
    
    recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    recommender.fit(ratings_df, movies_df)
    
    return recommender, ratings_df, movies_df


def load_user_history(user_id):
    history_file = Path(f'./user_history_{user_id}.json')
    if history_file.exists():
        with open(history_file, 'r') as f:
            return json.load(f)
    return {'liked': [], 'disliked': [], 'watched': [], 'ratings': {}}


def save_user_history(user_id, history):
    history_file = Path(f'./user_history_{user_id}.json')
    with open(history_file, 'w') as f:
        json.dump(history, f)


def load_all_users():
    users_file = Path('./all_users.json')
    if users_file.exists():
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}


def save_all_users(users):
    users_file = Path('./all_users.json')
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=2)


def get_genre_preference(user_history, movies_df):
    all_rated_movies = list(user_history['ratings'].keys())
    if not all_rated_movies:
        return {}
    
    genre_ratings = {}
    for movie_title, rating in user_history['ratings'].items():
        movie_data = movies_df[movies_df['title'] == movie_title]
        if not movie_data.empty:
            genres = str(movie_data['genres'].iloc[0]).split('|')
            for genre in genres:
                if genre not in genre_ratings:
                    genre_ratings[genre] = []
                genre_ratings[genre].append(float(rating))
    
    genre_preference = {}
    for genre, ratings in genre_ratings.items():
        genre_preference[genre] = np.mean(ratings)
    
    return dict(sorted(genre_preference.items(), key=lambda x: x[1], reverse=True))


def get_smart_recommendations(user_history, movies_df, n=10):
    if not user_history['ratings']:
        return movies_df.nlargest(n, 'popularity')
    
    genre_pref = get_genre_preference(user_history, movies_df)
    
    if not genre_pref:
        return movies_df.nlargest(n, 'popularity')
    
    top_genres = list(genre_pref.keys())[:5]
    
    if not top_genres:
        return movies_df.nlargest(n, 'popularity')
    
    genre_pattern = '|'.join(top_genres)
    
    filtered = movies_df[movies_df['genres'].str.contains(genre_pattern, na=False, regex=True)]
    
    watched_movies = set(user_history['watched'])
    filtered = filtered[~filtered['title'].isin(watched_movies)]
    
    if len(filtered) < n:
        filtered = movies_df[~movies_df['title'].isin(watched_movies)]
    
    return filtered.nlargest(n, 'popularity')


def calculate_match_score(movie_title, user_history, movies_df):
    if not user_history['ratings']:
        return 0.5
    
    movie_data = movies_df[movies_df['title'] == movie_title]
    if movie_data.empty:
        return 0.5
    
    movie_genres = str(movie_data['genres'].iloc[0]).split('|')
    genre_pref = get_genre_preference(user_history, movies_df)
    
    if not genre_pref:
        return 0.5
    
    scores = []
    for genre in movie_genres:
        if genre in genre_pref:
            scores.append(genre_pref[genre])
    
    if not scores:
        return 0.5
    
    match = np.mean(scores) / 5.0
    return min(match, 1.0)


recommender, ratings_df, movies_df = load_recommender()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.user_history = None

if not st.session_state.authenticated:
    st.title("Movie Recommendation System")
    st.markdown("Login or create a new account")
    
    auth_type = st.radio("Choose option:", ["Login", "Create Account"], horizontal=True)
    
    if auth_type == "Login":
        st.subheader("Login to Your Account")
        
        all_users = load_all_users()
        available_users = list(all_users.keys())
        
        if available_users:
            selected_user = st.selectbox("Select your account:", options=available_users)
            
            if st.button("Login", use_container_width=True):
                st.session_state.user_id = selected_user
                st.session_state.user_history = load_user_history(selected_user)
                st.session_state.authenticated = True
                st.success(f"Welcome back, {selected_user}!")
                st.rerun()
        else:
            st.info("No accounts yet. Create one first!")
            
            manual_user = st.text_input("Or enter your User ID:")
            if manual_user and st.button("Login with ID", use_container_width=True):
                st.session_state.user_id = manual_user
                st.session_state.user_history = load_user_history(manual_user)
                st.session_state.authenticated = True
                st.success(f"Welcome, {manual_user}!")
                st.rerun()
    
    else:
        st.subheader("Create New Account")
        
        new_user_id = st.text_input("Choose a User ID (username):", placeholder="e.g., anuja_123")
        
        if new_user_id and st.button("Create Account", use_container_width=True):
            all_users = load_all_users()
            
            if new_user_id in all_users:
                st.error("This User ID already exists!")
            else:
                all_users[new_user_id] = {
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'last_login': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                save_all_users(all_users)
                
                st.session_state.user_id = new_user_id
                st.session_state.user_history = load_user_history(new_user_id)
                st.session_state.authenticated = True
                
                st.success(f"Account created! Welcome, {new_user_id}!")
                st.rerun()

else:
    st.title("Movie Recommendation System")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown(f"**Logged in as: {st.session_state.user_id}**")
    
    with col3:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.user_history = None
            st.success("Logged out successfully!")
            st.rerun()
    
    st.markdown("Personalized movie recommendations with explainability")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Get Recommendations", "Smart Picks For You", "Your History", "Statistics"])
    
    with tab1:
        st.subheader("Find Movies You'll Love")
        
        search_type = st.radio(
            "How do you want to search?",
            ["By Movie Name", "By Genre", "Browse All"],
            horizontal=True
        )
        
        if search_type == "By Movie Name":
            st.markdown("**Enter a movie you like, and we'll recommend similar ones**")
            
            all_movies = sorted(movies_df['title'].unique())
            selected_movie = st.selectbox(
                "Select a movie you like",
                options=all_movies,
                key="movie_select"
            )
            
            if st.button("Get Recommendations", key="movie_button", use_container_width=True):
                st.session_state.show_recs = True
            
            if st.session_state.get('show_recs', False):
                movie_id = movies_df[movies_df['title'] == selected_movie]['movieId'].iloc[0]
                movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
                
                st.info(f"You selected: {selected_movie}")
                st.caption(f"Genres: {movie_info['genres']}")
                
                similar_movies = movies_df[
                    (movies_df['genres'].str.contains(movie_info['genres'].split('|')[0], na=False)) &
                    (movies_df['movieId'] != movie_id)
                ].nlargest(10, 'popularity')
                
                st.subheader("Movies Similar to Your Selection")
                
                for idx, (_, movie) in enumerate(similar_movies.iterrows(), 1):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{idx}. {movie['title']}**")
                        st.caption(movie['genres'])
                        if pd.notna(movie['overview']) and movie['overview']:
                            st.write(movie['overview'][:250] + "...")
                    
                    with col2:
                        st.metric("Popularity", f"{movie['popularity']:.1f}")
                    
                    with col3:
                        if movie['title'] in st.session_state.user_history['ratings']:
                            rating = st.session_state.user_history['ratings'][movie['title']]
                            st.metric("Your Rating", f"{rating}/5")
                        else:
                            st.metric("Not Rated", "-")
                    
                    col_like, col_dislike, col_watch, col_rate = st.columns(4)
                    with col_like:
                        if st.button("Like", key=f"like_{idx}_{selected_movie}"):
                            if movie['title'] not in st.session_state.user_history['liked']:
                                st.session_state.user_history['liked'].append(movie['title'])
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.success(f"Added to liked!")
                    
                    with col_dislike:
                        if st.button("Dislike", key=f"dislike_{idx}_{selected_movie}"):
                            if movie['title'] not in st.session_state.user_history['disliked']:
                                st.session_state.user_history['disliked'].append(movie['title'])
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.info(f"Added to disliked!")
                    
                    with col_watch:
                        if st.button("Watched", key=f"watch_{idx}_{selected_movie}"):
                            if movie['title'] not in st.session_state.user_history['watched']:
                                st.session_state.user_history['watched'].append(movie['title'])
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.success(f"Added to watched!")
                    
                    with col_rate:
                        rating = st.selectbox(
                            "Rate",
                            options=["-", "1", "2", "3", "4", "5"],
                            key=f"rating_{idx}_{selected_movie}"
                        )
                        if rating != "-":
                            st.session_state.user_history['ratings'][movie['title']] = int(rating)
                            save_user_history(st.session_state.user_id, st.session_state.user_history)
                            st.success(f"Rated {rating}/5")
                    
                    st.divider()
        
        elif search_type == "By Genre":
            st.markdown("**Select genres and we'll recommend popular movies in those categories**")
            
            all_genres = set()
            for genres in movies_df['genres'].dropna():
                all_genres.update(str(genres).split('|'))
            all_genres = sorted(list(all_genres))
            
            selected_genres = st.multiselect(
                "Select genres you like",
                options=all_genres,
                default=['Drama', 'Action']
            )
            
            if st.button("Get Recommendations", key="genre_button", use_container_width=True):
                st.session_state.show_genre_recs = True
            
            if st.session_state.get('show_genre_recs', False):
                if selected_genres:
                    genre_pattern = '|'.join(selected_genres)
                    filtered_movies = movies_df[
                        movies_df['genres'].str.contains(genre_pattern, na=False, regex=True)
                    ].nlargest(10, 'popularity')
                    
                    st.subheader(f"Top Movies in {', '.join(selected_genres)}")
                    
                    for idx, (_, movie) in enumerate(filtered_movies.iterrows(), 1):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{idx}. {movie['title']}**")
                            st.caption(movie['genres'])
                            if pd.notna(movie['overview']) and movie['overview']:
                                st.write(movie['overview'][:250] + "...")
                        
                        with col2:
                            st.metric("Popularity", f"{movie['popularity']:.1f}")
                        
                        with col3:
                            if movie['title'] in st.session_state.user_history['ratings']:
                                rating = st.session_state.user_history['ratings'][movie['title']]
                                st.metric("Your Rating", f"{rating}/5")
                            else:
                                st.metric("Not Rated", "-")
                        
                        col_like, col_dislike, col_watch, col_rate = st.columns(4)
                        with col_like:
                            if st.button("Like", key=f"like_genre_{idx}_{genre_pattern}"):
                                if movie['title'] not in st.session_state.user_history['liked']:
                                    st.session_state.user_history['liked'].append(movie['title'])
                                    save_user_history(st.session_state.user_id, st.session_state.user_history)
                                    st.success(f"Added to liked!")
                        
                        with col_dislike:
                            if st.button("Dislike", key=f"dislike_genre_{idx}_{genre_pattern}"):
                                if movie['title'] not in st.session_state.user_history['disliked']:
                                    st.session_state.user_history['disliked'].append(movie['title'])
                                    save_user_history(st.session_state.user_id, st.session_state.user_history)
                                    st.info(f"Added to disliked!")
                        
                        with col_watch:
                            if st.button("Watched", key=f"watch_genre_{idx}_{genre_pattern}"):
                                if movie['title'] not in st.session_state.user_history['watched']:
                                    st.session_state.user_history['watched'].append(movie['title'])
                                    save_user_history(st.session_state.user_id, st.session_state.user_history)
                                    st.success(f"Added to watched!")
                        
                        with col_rate:
                            rating = st.selectbox(
                                "Rate",
                                options=["-", "1", "2", "3", "4", "5"],
                                key=f"rating_genre_{idx}_{genre_pattern}"
                            )
                            if rating != "-":
                                st.session_state.user_history['ratings'][movie['title']] = int(rating)
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.success(f"Rated {rating}/5")
                        
                        st.divider()
                else:
                    st.warning("Please select at least one genre")
        
        else:
            st.markdown("**Browse all movies from 2020-2023**")
            
            selected_year = st.selectbox("Select year", options=[2023, 2022, 2021, 2020], key="year_select")
            
            display_movies = movies_df[movies_df['year'] == selected_year].nlargest(10, 'popularity')
            st.subheader(f"Top Movies in {int(selected_year)}")
            
            if len(display_movies) > 0:
                for idx, (_, movie) in enumerate(display_movies.iterrows(), 1):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        title_clean = movie['title'].split('(')[0].strip()
                        st.markdown(f"**{idx}. {title_clean} ({int(movie['year'])})**")
                        st.caption(movie['genres'])
                        if pd.notna(movie['overview']) and movie['overview']:
                            st.write(movie['overview'][:250] + "...")
                    
                    with col2:
                        st.metric("Popularity", f"{movie['popularity']:.1f}")
                    
                    with col3:
                        if movie['title'] in st.session_state.user_history['ratings']:
                            rating = st.session_state.user_history['ratings'][movie['title']]
                            st.metric("Your Rating", f"{rating}/5")
                        else:
                            st.metric("Not Rated", "-")
                    
                    col_like, col_dislike, col_watch, col_rate = st.columns(4)
                    with col_like:
                        if st.button("Like", key=f"like_browse_{idx}_{selected_year}"):
                            if movie['title'] not in st.session_state.user_history['liked']:
                                st.session_state.user_history['liked'].append(movie['title'])
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.success(f"Added to liked!")
                    
                    with col_dislike:
                        if st.button("Dislike", key=f"dislike_browse_{idx}_{selected_year}"):
                            if movie['title'] not in st.session_state.user_history['disliked']:
                                st.session_state.user_history['disliked'].append(movie['title'])
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.info(f"Added to disliked!")
                    
                    with col_watch:
                        if st.button("Watched", key=f"watch_browse_{idx}_{selected_year}"):
                            if movie['title'] not in st.session_state.user_history['watched']:
                                st.session_state.user_history['watched'].append(movie['title'])
                                save_user_history(st.session_state.user_id, st.session_state.user_history)
                                st.success(f"Added to watched!")
                    
                    with col_rate:
                        rating = st.selectbox(
                            "Rate",
                            options=["-", "1", "2", "3", "4", "5"],
                            key=f"rating_browse_{idx}_{selected_year}"
                        )
                        if rating != "-":
                            st.session_state.user_history['ratings'][movie['title']] = int(rating)
                            save_user_history(st.session_state.user_id, st.session_state.user_history)
                            st.success(f"Rated {rating}/5")
                    
                    st.divider()
            else:
                st.info(f"No movies found for {int(selected_year)}")
    
    with tab2:
        st.subheader("Smart Recommendations For You")
        
        st.session_state.user_history = load_user_history(st.session_state.user_id)
        
        if len(st.session_state.user_history['ratings']) == 0:
            st.info("Rate some movies first to get personalized recommendations!")
        else:
            genre_pref = get_genre_preference(st.session_state.user_history, movies_df)
            
            st.write(f"**Your Top Genres:** {', '.join(list(genre_pref.keys())[:3])}")
            
            if st.session_state.user_history['ratings']:
                avg_rating = np.mean(list(st.session_state.user_history['ratings'].values()))
                st.write(f"**Your Average Rating:** {avg_rating:.1f}/5")
            
            st.divider()
            
            smart_recs = get_smart_recommendations(st.session_state.user_history, movies_df, n=10)
            
            st.subheader("Based on Your Preferences")
            
            for idx, (_, movie) in enumerate(smart_recs.iterrows(), 1):
                match_score = calculate_match_score(movie['title'], st.session_state.user_history, movies_df)
                match_percent = int(match_score * 100)
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{idx}. {movie['title']}**")
                    st.caption(movie['genres'])
                    if pd.notna(movie['overview']) and movie['overview']:
                        st.write(movie['overview'][:250] + "...")
                
                with col2:
                    st.metric("Match", f"{match_percent}%")
                
                col_like, col_dislike, col_watch, col_rate = st.columns(4)
                with col_like:
                    if st.button("Like", key=f"like_smart_{idx}"):
                        if movie['title'] not in st.session_state.user_history['liked']:
                            st.session_state.user_history['liked'].append(movie['title'])
                            save_user_history(st.session_state.user_id, st.session_state.user_history)
                            st.success("Added to liked!")
                
                with col_dislike:
                    if st.button("Dislike", key=f"dislike_smart_{idx}"):
                        if movie['title'] not in st.session_state.user_history['disliked']:
                            st.session_state.user_history['disliked'].append(movie['title'])
                            save_user_history(st.session_state.user_id, st.session_state.user_history)
                            st.info("Added to disliked!")
                
                with col_watch:
                    if st.button("Watched", key=f"watch_smart_{idx}"):
                        if movie['title'] not in st.session_state.user_history['watched']:
                            st.session_state.user_history['watched'].append(movie['title'])
                            save_user_history(st.session_state.user_id, st.session_state.user_history)
                            st.success("Added to watched!")
                
                with col_rate:
                    rating = st.selectbox(
                        "Rate",
                        options=["-", "1", "2", "3", "4", "5"],
                        key=f"rating_smart_{idx}"
                    )
                    if rating != "-":
                        st.session_state.user_history['ratings'][movie['title']] = int(rating)
                        save_user_history(st.session_state.user_id, st.session_state.user_history)
                        st.success(f"Rated {rating}/5")
                
                st.divider()
    
    with tab3:
        st.subheader("Your Personalized Stats")
        
        st.session_state.user_history = load_user_history(st.session_state.user_id)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Movies Rated", len(st.session_state.user_history['ratings']))
        with col2:
            st.metric("Watched", len(st.session_state.user_history['watched']))
        with col3:
            st.metric("Liked", len(st.session_state.user_history['liked']))
        with col4:
            st.metric("Disliked", len(st.session_state.user_history['disliked']))
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Your Stats")
            
            if st.session_state.user_history['ratings']:
                avg_rating = np.mean(list(st.session_state.user_history['ratings'].values()))
                st.metric("Average Rating", f"{avg_rating:.2f}/5")
            else:
                st.write("No ratings yet")
            
            genre_pref = get_genre_preference(st.session_state.user_history, movies_df)
            if genre_pref:
                st.write("**Preferred Genres:**")
                for i, (genre, score) in enumerate(list(genre_pref.items())[:5], 1):
                    st.write(f"{i}. {genre}: {score:.2f}/5")
            else:
                st.write("Rate more movies to see genre preferences")
        
        with col2:
            st.subheader("Your History")
            
            col_liked, col_watched, col_disliked = st.columns(3)
            
            with col_liked:
                st.subheader("Liked")
                if st.session_state.user_history['liked']:
                    for movie in st.session_state.user_history['liked'][:5]:
                        st.write(f"- {movie}")
                    if len(st.session_state.user_history['liked']) > 5:
                        st.write(f"...+{len(st.session_state.user_history['liked']) - 5} more")
                else:
                    st.write("No liked movies")
            
            with col_watched:
                st.subheader("Watched")
                if st.session_state.user_history['watched']:
                    for movie in st.session_state.user_history['watched'][:5]:
                        st.write(f"- {movie}")
                    if len(st.session_state.user_history['watched']) > 5:
                        st.write(f"...+{len(st.session_state.user_history['watched']) - 5} more")
                else:
                    st.write("No watched movies")
            
            with col_disliked:
                st.subheader("Disliked")
                if st.session_state.user_history['disliked']:
                    for movie in st.session_state.user_history['disliked'][:5]:
                        st.write(f"- {movie}")
                    if len(st.session_state.user_history['disliked']) > 5:
                        st.write(f"...+{len(st.session_state.user_history['disliked']) - 5} more")
                else:
                    st.write("No disliked movies")
        
        st.divider()
        
        if st.button("Clear All History", use_container_width=True):
            st.session_state.user_history = {'liked': [], 'disliked': [], 'watched': [], 'ratings': {}}
            save_user_history(st.session_state.user_id, st.session_state.user_history)
            st.success("History cleared!")
    
    with tab4:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Movies", f"{len(movies_df):,}")
        
        with col2:
            st.metric("Total Ratings", f"{len(ratings_df):,}")
        
        with col3:
            st.metric("Year Range", "2020-2023")
        
        with col4:
            all_genres_set = set()
            for genres in movies_df['genres'].dropna():
                all_genres_set.update(str(genres).split('|'))
            st.metric("Genres", len(all_genres_set))
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Movies by Year")
            year_counts = movies_df['year'].value_counts().sort_index()
            st.bar_chart(year_counts)
        
        with col2:
            st.subheader("Top Genres")
            all_genres_dict = {}
            for genres in movies_df['genres'].dropna():
                for genre in str(genres).split('|'):
                    all_genres_dict[genre] = all_genres_dict.get(genre, 0) + 1
            
            top_genres = sorted(all_genres_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            genre_df = pd.DataFrame(top_genres, columns=['Genre', 'Count'])
            st.bar_chart(genre_df.set_index('Genre'))