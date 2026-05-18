import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOVIES_PATH = os.path.join(BASE_DIR, 'data', 'tmdb_5000_movies.csv')
CREDITS_PATH = os.path.join(BASE_DIR, 'data', 'tmdb_5000_credits.csv')
LINKS_PATH = os.path.join(BASE_DIR, 'data', 'links.csv')
RATINGS_PATH = os.path.join(BASE_DIR, 'data', 'ratings.csv')

movies = pd.read_csv(MOVIES_PATH)
credits = pd.read_csv(CREDITS_PATH)
links = pd.read_csv(LINKS_PATH)

dtypes = {
    'userId': 'int32',
    'movieId': 'int32',
    'rating': 'float32'
}
ratings = pd.read_csv(RATINGS_PATH, usecols=['userId', 'movieId', 'rating'], dtype=dtypes).sample(frac=0.2, random_state=42)

links.dropna(inplace=True)
links["tmdbId"] = links["tmdbId"].astype(int)
movies = movies.merge(credits, on = "title")
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies = movies.merge(links[['movieId', 'tmdbId']], left_on='movie_id', right_on='tmdbId', how='inner')
movies.drop(columns=['tmdbId'], inplace=True)

movies.rename(columns={"movie_id":"tmdbID"},inplace=True)

movies.dropna(inplace=True)

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x:x[0:10])

def fetch_director(text):
    List = []
    for crew in ast.literal_eval(text):
        if crew['job'] == 'Director':
            List.append(crew['name'])
    return List 
movies['crew'] = movies['crew'].apply(fetch_director)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
final_movies = movies.drop(columns=['overview','genres','keywords','cast','crew'])
final_movies['tags'] = final_movies['tags'].apply(lambda x: " ".join(x))
final_movies.head()

tf_idf = TfidfVectorizer(max_features=5000, stop_words='english')
movie_vectors = tf_idf.fit_transform(final_movies['tags'])
movie_vectors.shape

movie_vectors

similarity = cosine_similarity(movie_vectors)

similarity

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()
svd_model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
svd_model.fit(trainset)

def recommend(user_id, movie_title):
  if movie_title not in final_movies["title"].values:
    return []

  movie_idx = final_movies[final_movies["title"] == movie_title].index[0]
  scores = list(enumerate(similarity[movie_idx]))
  scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:101]

  hybrid_pred = []

  for idx, score in scores:
    row = final_movies.iloc[idx]
    tmdb_id = row["tmdbID"]
    movie_id = row["movieId"]
    title = row["title"]
    svd_pred = svd_model.predict(user_id, movie_id).est
    normalized_svd = (svd_pred - 0.5)/4.5
    final_score = (0.6 * score) + (0.4 * normalized_svd)

    hybrid_pred.append({
        "tmdbID": tmdb_id,
        "movieID": movie_id,
        "title": title,
        "score": final_score
    })

  hybrid_pred = sorted(hybrid_pred, key=lambda x: x["score"], reverse=True)
  return hybrid_pred[:5]

model_path = os.path.join(BASE_DIR, 'models')

if not os.path.exists(model_path):
    os.makedirs(model_path)
    print(f"Created directory: {model_path}")

pickle.dump(final_movies, open(os.path.join(model_path, 'movie_list.pkl'), 'wb'))
pickle.dump(similarity, open(os.path.join(model_path, 'similarity.pkl'), 'wb'))
pickle.dump(svd_model, open(os.path.join(model_path, 'svd_model.pkl'), 'wb'))
