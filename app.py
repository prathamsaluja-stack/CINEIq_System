import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(page_title="CINEIQ Recommender", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIE_LIST_PATH = os.path.join(BASE_DIR, 'models', 'movie_list.pkl')

def fetch_poster(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    fallback_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/500px-No_image_available.svg.png"
    
    try:
        data = requests.get(url, timeout=3).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return fallback_image
    except:
        return fallback_image

st.title('Movie Recommender')
st.markdown("---")

st.sidebar.header("User Settings")
user_id = st.sidebar.number_input("Enter Your User ID", min_value=1, value=1, step=1)
st.sidebar.info("Each user can have their own recommendation list")


@st.cache_resource
def load_data():
    import pickle
    movies = pickle.load(open(MOVIE_LIST_PATH, 'rb'))
    return movies

movies = load_data()
movie_list = movies['title'].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

if st.button('Recommend Movies'):

    backend_url = f"http://127.0.0.1:8000/recommend"
    params = {"user_id": user_id, "movie_title": selected_movie}
    
    with st.spinner('Finding Recommendations...'):
        try:
            response = requests.get(backend_url, params=params)
            
            if response.status_code == 200:
                recommendations = response.json()
                cols = st.columns(5)
                
                for idx, movie in enumerate(recommendations):
                    with cols[idx]:
                        poster_url = fetch_poster(movie['tmdbID'])
                        st.image(poster_url, width='stretch')
                     
                        st.markdown(f"**{movie['title']}**")
                        
                        st.metric(label="Audience Approval", value=movie.get('approval', 'N/A'))
                        st.caption(f"Match Score: {round(movie['score'] * 100, 1)}%")
                        
            else:
                st.error(f"Backend Error: {response.json().get('detail', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI backend. Is uvicorn running?")

st.sidebar.markdown("---")
show_dashboard = st.sidebar.checkbox("Show User Taste Dashboard")

if show_dashboard:
    st.header(f"User {user_id} Taste Profile")
    
    try:
        stats_res = requests.get(f"http://127.0.0.1:8000/user_stats/{user_id}")
        if stats_res.status_code == 200:
            stats = stats_res.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                categories = list(stats['genres'].keys())
                values = list(stats['genres'].values())
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Genre Affinity',
                    line_color='#00d4ff'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="white",
                    title="Genre Affinity Radar"
                )
                st.plotly_chart(fig_radar, width="stretch")

            with col2:
                decades = list(stats['decades'].keys())
                counts = list(stats['decades'].values())
                
                fig_bar = px.bar(
                    x=decades, y=counts, 
                    title="Movies Rated by Decade",
                    labels={'x': 'Decade', 'y': 'Count'},
                    color_discrete_sequence=['#00d4ff']
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color="white"
                )
                st.plotly_chart(fig_bar, width="stretch")
        else:
            st.warning("No rating history found for this User ID to generate a dashboard.")
    except:
        st.error("Could not load dashboard. Ensure Backend is running.")

st.markdown("---")
st.caption("CINEIQ Engine | Collaborative (SVD) + Content (TF-IDF) + VADER NLP")