import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import random
import urllib.parse

# Page config
st.set_page_config(
    page_title="CrossTune: Movie to Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# OMDb API setup (free, get key from http://www.omdbapi.com/apikey.aspx)
OMDB_API_KEY = st.secrets("OMDB_API_KEY")  # Get free key (1000 requests/day)

# Load models and data
@st.cache_resource
def load_models():
    try:
        vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
        movies_df = pd.read_csv('data/movies_processed.csv')
        tracks_df = pd.read_csv('data/tracks_processed.csv')
        
        # Pre-compute vectors
        movie_vectors = vectorizer.transform(movies_df['genres_expanded'])
        music_vectors = vectorizer.transform(tracks_df['tags_clean'])
        
        return vectorizer, movies_df, tracks_df, movie_vectors, music_vectors
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

vectorizer, movies_df, tracks_df, movie_vectors, music_vectors = load_models()

# Get movie poster from OMDb
@st.cache_data
def get_movie_poster(movie_title):
    """Fetch movie poster from OMDb API"""
    try:
        # Extract year and clean title
        if '(' in movie_title and ')' in movie_title:
            year_str = movie_title[movie_title.rfind('(')+1:movie_title.rfind(')')]
            movie_title_clean = movie_title[:movie_title.rfind('(')].strip()
            try:
                year = int(year_str)
            except:
                year = None
                movie_title_clean = movie_title
        else:
            movie_title_clean = movie_title
            year = None
        
        # Query OMDb
        params = {
            'apikey': OMDB_API_KEY,
            't': movie_title_clean
        }
        if year:
            params['y'] = year
        
        response = requests.get('http://www.omdbapi.com/', params=params, timeout=2)
        data = response.json()
        
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data['Poster']
    except:
        pass
    
    return "https://via.placeholder.com/300x450?text=No+Poster"

# Create Spotify search link
def create_spotify_link(track_name, artist_name):
    """Generate Spotify search URL for track and artist"""
    # Create search query: track name + artist name
    query = f"{track_name} {artist_name}"
    encoded_query = urllib.parse.quote(query)
    
    # Spotify search URL format
    spotify_url = f"https://open.spotify.com/search/{encoded_query}"
    
    return spotify_url

# Sample random movies
def get_random_movies(n=10):
    """Get n random movies from dataset"""
    return movies_df.sample(n=n)['title'].tolist()

# Create taste profile from selected movies
def create_taste_profile(selected_movie_titles):
    """Aggregate genre vectors from multiple movies to create taste profile"""
    movie_indices = [movies_df[movies_df['title'] == title].index[0] for title in selected_movie_titles]
    
    # Get vectors for selected movies
    selected_vectors = movie_vectors[movie_indices]
    
    # Aggregate: mean of all selected movie vectors
    # Convert sparse matrix mean to array (np.matrix is deprecated)
    taste_profile_vector = np.asarray(selected_vectors.mean(axis=0))
    
    return taste_profile_vector

# Recommend music based on taste profile
def recommend_music_from_profile(taste_profile_vector, n_recommendations=20):
    """Generate music recommendations from aggregated taste profile"""
    # Compute similarities
    similarities = cosine_similarity(taste_profile_vector, music_vectors)[0]
    
    # Get top N
    top_indices = similarities.argsort()[-n_recommendations:][::-1]
    
    # Build results
    results = tracks_df.iloc[top_indices][['name', 'artist', 'tags']].copy()
    results['similarity'] = similarities[top_indices]
    
    return results

# ========== SESSION STATE ==========
if 'movie_pool' not in st.session_state:
    st.session_state.movie_pool = get_random_movies(10)
if 'selected_movies' not in st.session_state:
    st.session_state.selected_movies = []

# ========== UI ==========

# Header
st.title("🎬🎵 CrossTune")
st.markdown("### Build your movie taste profile and discover your perfect playlist")
st.markdown("Select **3-5 movies** you love to get personalized music recommendations")
st.markdown("---")

if movies_df is not None:
    
    # Step 1: Movie Selection
    st.subheader("📽️ Step 1: Select Movies You Love")
    
    col_refresh, col_clear = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Movie Options", use_container_width=True):
            st.session_state.movie_pool = get_random_movies(10)
            st.rerun()
    with col_clear:
        if st.button("🗑️ Clear Selection", use_container_width=True):
            st.session_state.selected_movies = []
            st.rerun()
    
    st.markdown(f"**Selected: {len(st.session_state.selected_movies)} movies**")
    
    # Display movie grid
    cols = st.columns(5)
    
    for idx, movie_title in enumerate(st.session_state.movie_pool):
        col = cols[idx % 5]
        
        with col:
            # Get poster
            poster_url = get_movie_poster(movie_title)
            
            # Display poster
            st.image(poster_url, use_container_width=True)
            
            # Movie title (truncated)
            display_title = movie_title if len(movie_title) < 30 else movie_title[:27] + "..."
            st.markdown(f"<small>{display_title}</small>", unsafe_allow_html=True)
            
            # Selection button
            is_selected = movie_title in st.session_state.selected_movies
            
            if is_selected:
                if st.button("✅ Selected", key=f"btn_{idx}", use_container_width=True, type="primary"):
                    st.session_state.selected_movies.remove(movie_title)
                    st.rerun()
            else:
                if st.button("➕ Select", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.selected_movies.append(movie_title)
                    st.rerun()
    
    st.markdown("---")
    
    # Step 2: Generate Recommendations
    st.subheader("🎵 Step 2: Get Your Personalized Playlist")
    
    if len(st.session_state.selected_movies) >= 1:
        col_genre, col_gen = st.columns([2, 1])
        
        with col_genre:
            # Show combined genres
            selected_genres = []
            for movie in st.session_state.selected_movies:
                genres = movies_df[movies_df['title'] == movie].iloc[0]['genres']
                selected_genres.extend(genres.split('|'))
            
            unique_genres = list(set(selected_genres))
            st.info(f"**Your taste profile includes:** {', '.join(unique_genres)}")
        
        with col_gen:
            n_recs = st.slider("Number of songs:", 10, 30, 20)
        
        if st.button("🎵 Generate My Playlist", use_container_width=True, type="primary"):
            with st.spinner("Analyzing your movie taste and finding perfect music matches..."):
                # Create taste profile
                taste_profile = create_taste_profile(st.session_state.selected_movies)
                
                # Generate recommendations
                recommendations = recommend_music_from_profile(taste_profile, n_recommendations=n_recs)
            
            st.success(f"✨ Found {len(recommendations)} tracks that match your taste!")
            
            # Display recommendations in a nice format
            st.markdown("### 🎧 Your Personalized Playlist")
            
            for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    col_rank, col_info, col_score, col_spotify = st.columns([0.3, 2.5, 0.7, 0.8])
                    
                    with col_rank:
                        st.markdown(f"### {idx}")
                    
                    with col_info:
                        st.markdown(f"**{row['name']}**")
                        st.markdown(f"*{row['artist']}*")
                        st.caption(f"🏷️ {row['tags']}")
                    
                    with col_score:
                        # Match score
                        score_pct = row['similarity'] * 100
                        st.metric("Match", f"{score_pct:.0f}%")
                    
                    with col_spotify:
                        # Spotify link
                        spotify_url = create_spotify_link(row['name'], row['artist'])
                        st.link_button("🎧 Spotify", spotify_url, use_container_width=True)
                    
                    st.markdown("---")
    else:
        st.warning("👆 Please select at least 1 movie to get recommendations!")

else:
    st.error("⚠️ Could not load models. Please ensure all required files are present.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with ❤️ using Content-Based Cross-Domain Recommendation | Semantic Genre-to-Music Mapping with GloVe</p>
    <p>Movie Data: MovieLens & OMDb | Music Data: Last.fm | Streaming: Spotify</p>
</div>
""", unsafe_allow_html=True)


