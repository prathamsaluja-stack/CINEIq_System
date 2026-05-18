from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

load_dotenv()
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


ratings = pickle.load(open(os.path.join(BASE_DIR, 'models', 'user_ratings.pkl'), 'rb'))
MOVIE_LIST_PATH = os.path.join(BASE_DIR, 'models', 'movie_list.pkl')
SIMILARITY_PATH = os.path.join(BASE_DIR, 'models', 'similarity.pkl')
SVD_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'svd_model.pkl')

movies_df = pickle.load(open(MOVIE_LIST_PATH, 'rb'))
similarity = pickle.load(open(SIMILARITY_PATH, 'rb'))
svd_model = pickle.load(open(SVD_MODEL_PATH, 'rb'))

sia = SentimentIntensityAnalyzer()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_reviews(tmdb_id: int):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code != 200: return []
        data = response.json()
        return [result['content'][:1000] for result in data.get('results', [])[:5]]
    except:
        return []

def fetch_fallback_score(tmdb_id: int):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url, timeout=3).json()
        return data.get('vote_average', 5.0) / 10.0 
    except:
        return 0.5

@app.get("/recommend")
def get_recommendations(user_id: int, movie_title: str):
    if movie_title not in movies_df["title"].values:
        raise HTTPException(status_code=404, detail="Movie not found")

    movie_idx = movies_df[movies_df["title"] == movie_title].index[0]
    scores = sorted(list(enumerate(similarity[movie_idx])), key=lambda x: x[1], reverse=True)[1:51]

    hybrid_pred = []
    for idx, content_score in scores:
        row = movies_df.iloc[idx]
        svd_pred = svd_model.predict(user_id, row["movieId"]).est 
        final_score = (0.6 * content_score) + (0.4 * (svd_pred - 0.5)/4.5)
        
        hybrid_pred.append({
            "tmdbID": int(row["tmdbID"]),
            "title": row["title"],
            "score": float(final_score) 
        })

    final_results = []
    for movie in sorted(hybrid_pred, key=lambda x: x["score"], reverse=True)[:5]:
        reviews = fetch_reviews(movie["tmdbID"])
        if not reviews:
            fallback_pos = fetch_fallback_score(movie["tmdbID"])
            movie["sentiment_score"] = float(movie["score"] * fallback_pos)
            movie["approval"] = f"{round(fallback_pos * 100)}% (TMDB Avg)"
            
        else:
            total_pos_score = 0
            for review in reviews:
                vader_scores = sia.polarity_scores(review)
                compound = vader_scores['compound']
                
                normalized_pos = (compound + 1) / 2 
                total_pos_score += normalized_pos
                
            avg_pos = total_pos_score / len(reviews)
            
            movie["sentiment_score"] = float(movie["score"] * avg_pos)
            movie["approval"] = f"{round(avg_pos * 100)}%"
            
        final_results.append(movie)

    return sorted(final_results, key=lambda x: x["sentiment_score"], reverse=True)[:5]

@app.get("/user_stats/{user_id}")
def get_user_stats(user_id: int):
    user_ratings = ratings[ratings['userId'] == user_id]
    
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail="User history not found")

    user_history = user_ratings.merge(movies_df[['movieId', 'title', 'tmdbID']], on='movieId')
    
    high_rated = user_history[user_history['rating'] >= 4.0]
    
    genre_data = {
        "Action": 85, "Sci-Fi": 90, "Drama": 40, "Comedy": 30, "Thriller": 70
    }
    
    decade_data = {
        "1990s": 5, "2000s": 12, "2010s": 25, "2020s": 8
    }

    return {
        "genres": genre_data,
        "decades": decade_data,
        "total_ratings": len(user_ratings)
    }