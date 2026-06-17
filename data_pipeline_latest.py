import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path


class DataPipelineLatest:
    def __init__(self, data_dir='./data', min_year=2020, max_year=2026):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.min_year = min_year
        self.max_year = max_year
        
    def load_movielens_latest(self):
        print(f"Loading MovieLens Latest...")
        
        url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
        zip_path = self.data_dir / 'ml-latest.zip'
        extract_dir = self.data_dir / 'ml-latest'
        
        if not extract_dir.exists():
            print(f"  Downloading MovieLens data...")
            import urllib.request
            import zipfile
            
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print("  ✓ Downloaded and extracted")
        
        ratings_path = extract_dir / 'ratings.csv'
        movies_path = extract_dir / 'movies.csv'
        
        ratings_df = pd.read_csv(ratings_path)
        ratings_df = ratings_df[['userId', 'movieId', 'rating']].copy()
        
        movies_df = pd.read_csv(movies_path)
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')[0].astype(float)
        
        movies_df['overview'] = ''
        movies_df['popularity'] = 5.0
        movies_df = movies_df[['movieId', 'title', 'genres', 'year', 'overview', 'popularity']]
        
        print(f"  ✓ Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies")
        
        return ratings_df, movies_df
    
    def filter_recent_movies(self, ratings_df, movies_df):
        print(f"\nFiltering movies ({self.min_year}-{self.max_year})...")
        
        initial_movies = len(movies_df)
        initial_ratings = len(ratings_df)
        
        movies_df = movies_df.dropna(subset=['year'])
        movies_df = movies_df[
            (movies_df['year'] >= self.min_year) & 
            (movies_df['year'] <= self.max_year)
        ]
        
        recent_movie_ids = set(movies_df['movieId'])
        ratings_df = ratings_df[ratings_df['movieId'].isin(recent_movie_ids)]
        
        print(f"  Movies: {initial_movies:,} → {len(movies_df):,}")
        print(f"  Ratings: {initial_ratings:,} → {len(ratings_df):,}")
        
        return ratings_df, movies_df
    
    def enrich_with_tmdb_data(self, movies_df, tmdb_api_key=None, sample_size=None):
        import os
        
        if tmdb_api_key is None:
            tmdb_api_key = os.getenv('TMDB_API_KEY')
        
        if not tmdb_api_key:
            print("\n⚠ TMDB API key not found. Using default data.")
            return movies_df
        
        print("\nEnriching with TMDB data...")
        
        movies_df = movies_df.copy()
        
        if sample_size:
            movies_df_to_fetch = movies_df.head(sample_size)
        else:
            movies_df_to_fetch = movies_df
        
        base_url = 'https://api.themoviedb.org/3/search/movie'
        
        for idx, row in movies_df_to_fetch.iterrows():
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(movies_df_to_fetch)}...")
                time.sleep(0.5)
            
            try:
                title = row['title'].split('(')[0].strip()
                
                params = {
                    'api_key': tmdb_api_key,
                    'query': title,
                    'year': int(row['year']) if pd.notna(row['year']) else None
                }
                
                response = requests.get(base_url, params=params, timeout=5)
                results = response.json().get('results', [])
                
                if results:
                    movie = results[0]
                    overview = movie.get('overview', '')
                    popularity = movie.get('popularity', 5.0)
                else:
                    overview = ''
                    popularity = 5.0
                
            except Exception as e:
                overview = ''
                popularity = 5.0
            
            movies_df.loc[idx, 'overview'] = overview
            movies_df.loc[idx, 'popularity'] = popularity
        
        print("  ✓ TMDB enrichment complete")
        return movies_df
    
    def preprocess_data(self, ratings_df, movies_df, min_user_ratings=5, min_movie_ratings=2):
        print("\nPreprocessing data...")
        
        user_counts = ratings_df['userId'].value_counts()
        active_users = user_counts[user_counts >= min_user_ratings].index
        ratings_df = ratings_df[ratings_df['userId'].isin(active_users)].copy()
        
        movie_counts = ratings_df['movieId'].value_counts()
        popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(popular_movies)].copy()
        
        movies_df = movies_df[movies_df['movieId'].isin(ratings_df['movieId'].unique())].copy()
        
        movies_df['overview'] = movies_df['overview'].fillna('')
        movies_df['popularity'] = movies_df['popularity'].fillna(5.0)
        
        print(f"  ✓ {len(ratings_df):,} ratings, {len(movies_df):,} movies")
        print(f"    {ratings_df['userId'].nunique():,} active users")
        
        return ratings_df, movies_df
    
    def save_data(self, ratings_df, movies_df, filename_prefix='recent_movies'):
        ratings_path = self.data_dir / f'{filename_prefix}_ratings.csv'
        movies_path = self.data_dir / f'{filename_prefix}_movies.csv'
        
        ratings_df.to_csv(ratings_path, index=False)
        movies_df.to_csv(movies_path, index=False)
        
        print(f"\n✓ Data saved to {self.data_dir}/")
        
        return ratings_path, movies_path
    
    def load_processed_data(self, filename_prefix='recent_movies'):
        ratings_path = self.data_dir / f'{filename_prefix}_ratings.csv'
        movies_path = self.data_dir / f'{filename_prefix}_movies.csv'
        
        if not ratings_path.exists() or not movies_path.exists():
            raise FileNotFoundError(f"Processed data not found in {self.data_dir}/")
        
        ratings_df = pd.read_csv(ratings_path)
        movies_df = pd.read_csv(movies_path)
        
        print(f"✓ Loaded {len(ratings_df):,} ratings, {len(movies_df):,} movies")
        
        return ratings_df, movies_df