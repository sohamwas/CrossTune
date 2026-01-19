# CrossTune 🎬🎵  
*A Content-Based Cross-Domain Movie-to-Music Recommender*

CrossTune is a **content-based cross-domain recommendation system** that takes a user’s movie taste and recommends music tracks that match the **mood, genre, and vibe** of those movies.  
Instead of relying on overlapping users between domains, it bridges the gap using **semantic similarity** between movie genres and music tags.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-red)
![ML](https://img.shields.io/badge/ML-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-GloVe-green)
![Embeddings](https://img.shields.io/badge/Embeddings-TF--IDF-blue)
![API](https://img.shields.io/badge/API-OMDb-yellow)
![Music](https://img.shields.io/badge/Music-Spotify-brightgreen)
![Data](https://img.shields.io/badge/Data-MovieLens%20%7C%20Last.fm-purple)


**Core Libraries & Tools**
- **Python** – Core language for data processing and modeling  
- **Pandas, NumPy, SciPy** – Data manipulation, sparse matrices, numerical ops  
- **Scikit-learn** – TF-IDF vectorization, cosine similarity  
- **Gensim (GloVe embeddings)** – Semantic mapping between movie genres and music tags  
- **Implicit (ALS)** – Collaborative filtering embeddings (future hybrid extension)  
- **Streamlit** – Interactive web app for building taste profiles and serving recommendations  
- **Requests** – OMDb API integration for fetching movie posters  
- **Spotify (search URLs)** – “Listen on Spotify” links for recommended tracks  

---

## ✨ Features

### 🎯 Core Functionality

#### 🎥 Movie-to-Music Cross-Domain Recommendation
- User selects multiple movies they like (**no ratings needed**).
- System builds a **taste profile** by aggregating semantic genre representations.
- Recommends music tracks whose tags best align with that profile.

#### 🧠 Semantic Genre–Tag Mapping
- Movie genres (e.g., *Action, Romance, Thriller*) are expanded using **GloVe embeddings**.
- Music tags (e.g., *rock, indie, electronic, acoustic*) are vectorized with **TF-IDF**.
- Similarity is computed using **cosine similarity**.

---

## 🖥️ Streamlit App

### 🎬 Interactive Movie Selection
- Displays a grid of movies with posters (via **OMDb API**).
- Users can:
  - Refresh the movie pool  
  - Select / deselect multiple movies  
  - Build a custom movie taste profile  

### 🎶 Personalized Playlist Generation
- Aggregates selected movies’ genre vectors into a single **taste profile**.
- Generates a **ranked list** of music recommendations.

### 🎧 Spotify Integration
Each recommended track includes:
- Track name  
- Artist  
- Tags  
- **“Listen on Spotify”** button (opens a Spotify search for *track + artist*)

---

## 🎥 Demo

https://github.com/user-attachments/assets/1b42099a-1445-4181-af97-4b5f32692c5c

---

## ⚠️ Limitations

### Vocabulary Gap Between Domains
- Movie genres and music tags differ heavily (e.g., *Action* vs *Rock*).
- Semantic mapping via GloVe is an approximation and can yield unintuitive matches.

### Bias Toward Popular Tags
- Generic tags (*rock, pop, love*) dominate recommendations.
- Can reduce diversity and catalog coverage.

---

## ⚙️ Optimizations & Design Choices

### 🚀 On-Demand Similarity Computation
- Avoids precomputing a full **movie × track similarity matrix** (would require tens of GBs).
- Similarity is computed **on demand** for:
  - A single movie, or  
  - An aggregated taste profile  
- Keeps memory usage low and responsiveness high.

### 🗂️ Pre-Computed Semantic Mapping Cache
- Semantic similarities between **unique movie genres** and **unique music tag words** are:
  - Computed once  
  - Cached in `genre_to_music_map`
- Reduces billions of comparisons to a small reusable mapping.

### 📐 Mean Vector Taste Profile
- Uses the **mean of TF-IDF genre vectors** for multiple selected movies.
- Simple, efficient, and captures shared semantic structure.

---

## 🙏 Acknowledgments

### 🎬 Movie Data
- **MovieLens** – Movie ratings and metadata

### 🎵 Music Data
- **Last.fm** – Track-level listening histories and music tags

### 🧠 Semantic Embeddings
- **GloVe** – Pre-trained word embeddings for genre–tag alignment

### 🌐 APIs
- **OMDb API** – Movie details and posters  
- **Spotify** – “Listen on Spotify” search integration

### 📚 Libraries
NumPy, Pandas, SciPy, Scikit-learn, Gensim, Implicit, Streamlit, and the broader open-source ecosystem that made this project possible.
