# Movie Recommendation System

A production-ready hybrid recommendation system built with Python and Streamlit that provides personalized movie suggestions using collaborative filtering (60%) and content-based filtering (40%) algorithms.

## Live Demo

🎬 **[Try the Live App](https://movie-recommendation-1103.streamlit.app/)**

## Quick Overview

This project demonstrates a complete machine learning pipeline with a web interface, combining two recommendation algorithms to provide personalized movie suggestions based on user ratings and preferences.

**Dataset:** 4,908 movies | **Ratings:** 231,462 | **Time Period:** 2020-2023 | **Genres:** 17 categories

---

## Features

### 🔐 User Authentication
- Create account with persistent login
- Netflix-style account management
- User data saved locally
- Logout functionality

### 🎬 Movie Discovery
- Search by movie name
- Filter by multiple genres
- Browse by year (2020-2023)
- View movie details (genres, plot, popularity)

### ⭐ Rating System
- Rate movies 1-5 stars
- Like/Dislike movies
- Track watched movies
- Auto-save all interactions

### 🤖 Smart Recommendations
- AI-powered personalized suggestions
- Shows match score (0-100%)
- Learns your genre preferences
- Excludes already-watched movies

### 📊 Personalized Dashboard
- View your stats (ratings, average score)
- See favorite genres ranked by rating
- Complete history tracking
- Clear history option

### 📈 Dataset Analytics
- Movies by year distribution
- Genre popularity charts
- Dataset statistics
- Real-time data visualization

---

## Algorithm

### Hybrid Recommendation Engine

**Collaborative Filtering (60%)**
- Finds users with similar rating patterns
- Recommends highly-rated movies from similar users
- Uses cosine similarity on user-movie matrix

**Content-Based Filtering (40%)**
- Extracts movie features (genres, plot text)
- Uses TF-IDF vectorization
- Finds movies similar to ones you rated highly

**Final Score:** `(CF × 0.6) + (CB × 0.4)` normalized to 0-100%

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Framework** | Streamlit |
| **ML Libraries** | Scikit-learn, Pandas, NumPy |
| **Data** | MovieLens Latest, TMDB API |
| **Deployment** | Streamlit Cloud, GitHub |

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Windows (with flag for pandas)
pip install pandas numpy scikit-learn streamlit requests --no-build-isolation

# macOS/Linux
pip install -r requirements.txt
```

### 4. Optional: Set TMDB API Key
```bash
# Windows
$env:TMDB_API_KEY = 'your_api_key_here'

# macOS/Linux
export TMDB_API_KEY='your_api_key_here'
```

Get free API key from [TMDB](https://www.themoviedb.org/settings/api)

---

## Usage

### First Time: Train the Model
```bash
python train_latest.py train
```
- Downloads MovieLens dataset (~600MB)
- Filters to 2020-2023 movies
- Enriches with TMDB data
- Trains hybrid recommender
- **Time:** 10-15 minutes

### Run the Application
```bash
streamlit run app_production.py
```
Opens at: `http://localhost:8501`

### Using the App

**Tab 1: Get Recommendations**
- Search by movie name
- Filter by genre
- Browse by year

**Tab 2: Smart Picks For You**
- AI recommendations based on YOUR ratings
- Shows match percentage
- Learns your preferences

**Tab 3: Your History**
- View all your stats
- See genre preferences
- Check liked/watched/disliked lists

**Tab 4: Statistics**
- Dataset overview
- Charts by year & genre
- Movie & rating counts

---

## Project Structure

```
movie-recommendation-system/
├── app_production.py              # Main Streamlit app
├── recommendation_engine.py       # ML recommender class
├── data_pipeline_latest.py        # Data processing
├── train_latest.py                # Training script
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── .gitignore                     # Git exclusions
│
├── user_history_{user_id}.json    # User data (auto-created)
├── all_users.json                 # User registry (auto-created)
│
└── data/                          # MovieLens data (auto-created)
    ├── recent_movies_ratings.csv
    └── recent_movies_movies.csv
```

---

## Dataset

### MovieLens Latest
- **Source:** [GroupLens](https://grouplens.org/datasets/movielens/latest/)
- **Total Ratings:** 33,832,162
- **Total Movies:** 86,537
- **Filtered to:** 4,908 movies (2020-2023)
- **Active Ratings:** 231,462
- **Genres:** 17 categories

### TMDB Enrichment
Each movie includes:
- Genres
- Plot summary
- Popularity score
- Release year

---

## How It Works

### Data Pipeline
```
MovieLens (33M ratings) 
    → Filter (2020-2023) 
    → Enrich (TMDB API) 
    → Preprocess 
    → Train Model
    → Streamlit Interface
```

### Recommendation Flow
```
User Rates Movies 
    → Calculate Genre Preferences 
    → Collaborative Filtering 
    → Content-Based Filtering 
    → Combine Scores (60% + 40%) 
    → Rank & Return Top 10
```

---

## Performance

| Metric | Value |
|--------|-------|
| Training Time | 10-15 minutes |
| Inference Time | <1 second |
| Model Size | ~50MB |
| Data Sparsity | ~98% (typical) |
| Supported Users | Unlimited |

---

## Key Files Explained

### `app_production.py` (~900 lines)
Main Streamlit application featuring:
- Login/Registration system
- 4 functional tabs
- User history tracking
- Real-time recommendations
- Data visualization

### `recommendation_engine.py` (~250 lines)
Hybrid recommender class:
- `fit()` - Train on ratings data
- `recommend()` - Generate recommendations
- Collaborative filtering implementation
- Content-based filtering with TF-IDF

### `data_pipeline_latest.py` (~300 lines)
Data processing pipeline:
- MovieLens data loading
- TMDB API enrichment
- Data filtering & preprocessing
- Data persistence

### `train_latest.py` (~150 lines)
Training orchestration:
- Download MovieLens data
- Execute data pipeline
- Train recommender
- Save model & data

---

## Troubleshooting

### "ModuleNotFoundError"
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt --no-build-isolation
```

### "pandas build failed" (Windows)
```bash
pip install pandas --no-build-isolation
```

### Model not found
```bash
# Retrain the model
python train_latest.py train
```

### Ratings not saving
- Check username (no spaces/special chars)
- Ensure write permissions
- Try logging out and back in

### App crashes on deploy
- Verify `requirements.txt` in root
- Check `data/` in `.gitignore`
- Test locally first

---

## Deployment

### Deploy to Streamlit Cloud (FREE)

1. **Push code to GitHub** (see below)
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click "Sign in with GitHub"
4. Click "New app"
5. Select your repository
6. Set main file: `app_production.py`
7. Click "Deploy"
8. Wait 2-5 minutes
9. **Get your live URL!**

### Live App URL Format
```
https://share.streamlit.io/YOUR_USERNAME/movie-recommendation-system/main/app_production.py
```

---

## Portfolio Value

This project demonstrates expertise in:

### Machine Learning
✓ Hybrid recommender system design
✓ Collaborative filtering algorithms
✓ Content-based filtering (TF-IDF)
✓ Matrix operations & similarity metrics
✓ Feature engineering

### Data Engineering
✓ End-to-end ML pipeline
✓ API integration (TMDB)
✓ Large dataset handling (250K+ ratings)
✓ Data preprocessing & cleaning
✓ Sparse matrix operations

### Web Development
✓ Streamlit framework
✓ User authentication
✓ Responsive UI design
✓ Real-time data updates
✓ Multi-tab dashboard

### Software Engineering
✓ Production-ready code
✓ Error handling
✓ Code organization
✓ Git version control
✓ Complete documentation

---

## Use on Resume

### Data Science Track
```
Movie Recommendation System | GitHub | Live Demo
- Built hybrid recommender (60% CF + 40% CB)
- Processed 250K+ user ratings with scikit-learn
- Deployed on Streamlit Cloud with persistent auth
```

### Full-Stack Track
```
Movie Recommendation System | GitHub | Live Demo
- End-to-end ML pipeline (data → model → interface)
- Implemented user authentication system
- Deployed web app to production
```

### Data Engineering Track
```
Movie Recommendation System | GitHub | Live Demo
- Built data pipeline: MovieLens → TMDB enrichment
- Handled sparse matrix operations (98% sparsity)
- Integrated external APIs with rate limiting
```

---

## Requirements

```
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
streamlit==1.28.0
requests==2.31.0
```

---

## Future Enhancements

- [ ] Deep learning models (neural CF)
- [ ] Real-time model updates
- [ ] Social recommendations
- [ ] Movie watchlist with priorities
- [ ] Advanced search filters
- [ ] Email notifications
- [ ] Mobile app
- [ ] User profiles & images

---

## License

This project uses:
- **MovieLens Dataset** - Academic/research use
- **TMDB API** - Free developer license

---

## Author

**Anuja Patade*

---

## Acknowledgments

- [GroupLens](https://grouplens.org/) - MovieLens dataset
- [TMDB](https://www.themoviedb.org/) - Movie metadata API
- [Streamlit](https://streamlit.io/) - Web framework
- [Scikit-learn](https://scikit-learn.org/) - ML algorithms

---

## Quick Links

| Link | URL |
|------|-----|
| **Live App** | https://share.streamlit.io/YOUR_USERNAME/movie-recommendation-system/main/app_production.py |
| **MovieLens Dataset** | https://grouplens.org/datasets/movielens/latest/ |
| **TMDB API** | https://www.themoviedb.org/settings/api |
| **Streamlit Docs** | https://docs.streamlit.io/ |

---

## Getting Started

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system

# 2. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install
pip install -r requirements.txt --no-build-isolation

# 4. Train (first time only)
python train_latest.py train

# 5. Run
streamlit run app_production.py

# 6. Deploy (optional)
# Push to GitHub, then deploy on Streamlit Cloud
```
