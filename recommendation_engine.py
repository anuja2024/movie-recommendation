import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.ratings_df = None
        self.movies_df = None
        self.user_similarity = None
        self.content_similarity = None
        self.scaler = MinMaxScaler()
        self.user_item_matrix = None
        
    def fit(self, ratings_df, movies_df):
        print("Training hybrid recommender...")
        
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        print("  Computing user similarity...")
        user_item_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        self.user_similarity = pd.DataFrame(
            cosine_similarity(user_item_matrix),
            index=user_item_matrix.index,
            columns=user_item_matrix.index
        )
        self.user_item_matrix = user_item_matrix
        
        print("  Computing content similarity...")
        self._compute_content_similarity()
        
        print("✓ Model trained successfully")
        
    def _compute_content_similarity(self):
        self.movies_df['content'] = (
            self.movies_df['genres'].fillna('') + ' ' + 
            self.movies_df['overview'].fillna('')
        )
        
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['content'])
        
        content_sim = cosine_similarity(tfidf_matrix)
        self.content_similarity = pd.DataFrame(
            content_sim,
            index=self.movies_df['movieId'],
            columns=self.movies_df['movieId']
        )
        
    def collaborative_filtering(self, user_id, n_recommendations=10):
        if user_id not in self.user_similarity.index:
            return pd.DataFrame()
        
        similar_users = self.user_similarity[user_id].drop(user_id).nlargest(10)
        user_rated_movies = set(
            self.ratings_df[self.ratings_df['userId'] == user_id]['movieId']
        )
        
        recommendations = {}
        for similar_user, similarity_score in similar_users.items():
            similar_user_ratings = self.ratings_df[
                (self.ratings_df['userId'] == similar_user) &
                (~self.ratings_df['movieId'].isin(user_rated_movies))
            ]
            
            for _, row in similar_user_ratings.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                weighted_score = rating * similarity_score
                
                if movie_id not in recommendations:
                    recommendations[movie_id] = []
                recommendations[movie_id].append(weighted_score)
        
        cf_scores = {
            movie_id: np.mean(scores)
            for movie_id, scores in recommendations.items()
        }
        
        cf_df = pd.DataFrame(
            list(cf_scores.items()),
            columns=['movieId', 'cf_score']
        ).nlargest(n_recommendations * 2, 'cf_score')
        
        return cf_df
    
    def content_based_filtering(self, user_id, n_recommendations=10):
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        high_rated_movies = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
        
        if not high_rated_movies:
            return pd.DataFrame()
        
        recommendations = {}
        for movie_id in high_rated_movies:
            if movie_id not in self.content_similarity.index:
                continue
                
            similar_movies = self.content_similarity[movie_id].nlargest(20)
            
            for similar_movie_id, similarity_score in similar_movies.items():
                if similar_movie_id not in user_ratings['movieId'].values:
                    if similar_movie_id not in recommendations:
                        recommendations[similar_movie_id] = []
                    recommendations[similar_movie_id].append(similarity_score)
        
        cb_scores = {
            movie_id: np.mean(scores)
            for movie_id, scores in recommendations.items()
        }
        
        cb_df = pd.DataFrame(
            list(cb_scores.items()),
            columns=['movieId', 'cb_score']
        ).nlargest(n_recommendations * 2, 'cb_score')
        
        return cb_df
    
    def recommend(self, user_id, n_recommendations=10):
        cf_recs = self.collaborative_filtering(user_id, n_recommendations * 2)
        cb_recs = self.content_based_filtering(user_id, n_recommendations * 2)
        
        if cf_recs.empty and cb_recs.empty:
            return self._get_popular_movies(n_recommendations)
        
        all_recs = pd.DataFrame()
        
        if not cf_recs.empty:
            all_recs = cf_recs.copy()
        
        if not cb_recs.empty:
            if all_recs.empty:
                all_recs = cb_recs.copy()
            else:
                all_recs = all_recs.merge(cb_recs, on='movieId', how='outer').fillna(0)
        
        if 'cf_score' in all_recs.columns:
            all_recs['cf_score'] = self.scaler.fit_transform(all_recs[['cf_score']])
        else:
            all_recs['cf_score'] = 0
            
        if 'cb_score' in all_recs.columns:
            all_recs['cb_score'] = self.scaler.fit_transform(all_recs[['cb_score']])
        else:
            all_recs['cb_score'] = 0
        
        all_recs['hybrid_score'] = (
            (all_recs['cf_score'] * self.cf_weight) +
            (all_recs['cb_score'] * self.cb_weight)
        )
        
        top_recs = all_recs.nlargest(n_recommendations, 'hybrid_score')
        
        result = top_recs.merge(
            self.movies_df[['movieId', 'title', 'genres', 'overview', 'popularity']],
            on='movieId',
            how='left'
        )
        
        return result[['movieId', 'title', 'genres', 'overview', 'popularity',
                       'cf_score', 'cb_score', 'hybrid_score']].reset_index(drop=True)
    
    def _get_popular_movies(self, n=10):
        popular = self.movies_df.nlargest(n, 'popularity')[
            ['movieId', 'title', 'genres', 'overview', 'popularity']
        ].reset_index(drop=True)
        popular['hybrid_score'] = popular['popularity'] / popular['popularity'].max()
        return popular
    
    def explain_recommendation(self, user_id, movie_id):
        explanation = {}
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        similar_users = self.user_similarity[user_id].nlargest(5).index.tolist()
        similar_user_ratings = self.ratings_df[
            (self.ratings_df['userId'].isin(similar_users[1:])) &
            (self.ratings_df['movieId'] == movie_id)
        ]
        
        if not similar_user_ratings.empty:
            explanation['cf_reason'] = (
                f"Similar users rated {similar_user_ratings['rating'].mean():.1f}/5"
            )
        
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]['movieId'].tolist()
        if high_rated:
            recommended_genres = self.movies_df[
                self.movies_df['movieId'] == movie_id
            ]['genres'].iloc[0] if movie_id in self.movies_df['movieId'].values else ""
            
            if recommended_genres:
                explanation['cb_reason'] = f"Similar to movies you liked ({recommended_genres})"
        
        return explanation