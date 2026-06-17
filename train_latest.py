import sys
from pathlib import Path
import pandas as pd
import pickle

from data_pipeline_latest import DataPipelineLatest
from recommendation_engine import HybridRecommender


def main():
    print("\n" + "="*70)
    print("  MOVIE RECOMMENDATION SYSTEM - TRAINING")
    print("="*70 + "\n")
    
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    
    pipeline = DataPipelineLatest(data_dir=str(data_dir), min_year=2020, max_year=2026)
    
    ratings_df, movies_df = pipeline.load_movielens_latest()
    ratings_df, movies_df = pipeline.filter_recent_movies(ratings_df, movies_df)
    
    import os
    if os.getenv('TMDB_API_KEY'):
        print("\n✓ TMDB API key found")
        movies_df = pipeline.enrich_with_tmdb_data(movies_df)
    
    ratings_df, movies_df = pipeline.preprocess_data(ratings_df, movies_df)
    pipeline.save_data(ratings_df, movies_df)
    
    print("\nTraining recommender...")
    recommender = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    recommender.fit(ratings_df, movies_df)
    
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'hybrid_recommender_latest.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(recommender, f)
    
    print(f"\n✓ Model saved to {model_path}")
    
    print("\n" + "="*70)
    print("  ✓ TRAINING COMPLETE")
    print("="*70)
    print("\nNext: streamlit run app_production.py\n")


def test_recommendations(user_id=None, n_recs=10):
    model_path = Path('./models/hybrid_recommender_latest.pkl')
    
    if not model_path.exists():
        print("❌ Model not found. Run training first.")
        return
    
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)
    
    pipeline = DataPipelineLatest(data_dir='./data', min_year=2020, max_year=2026)
    ratings_df, movies_df = pipeline.load_processed_data()
    
    if user_id is None:
        user_id = ratings_df['userId'].sample(1).iloc[0]
    
    print(f"\n🎬 Recommendations for User {user_id}:\n")
    recommendations = recommender.recommend(user_id, n_recommendations=n_recs)
    
    for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
        print(f"{idx}. {rec['title']}")
        print(f"   Score: {rec['hybrid_score']:.3f}\n")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            main()
        elif command == 'test':
            user_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            test_recommendations(user_id=user_id)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python train_latest.py train|test [user_id]")
    else:
        main()