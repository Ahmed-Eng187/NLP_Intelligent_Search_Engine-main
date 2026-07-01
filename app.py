import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(page_title="Intelligent Search Engine", layout="wide")

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_resources():
    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    
    # Load dataset
    df = pd.read_csv('marketing_sample_for_flipkart_com-ecommerce__20191101_20191130__15k_data.csv', 
                     on_bad_lines='warn', encoding='ISO-8859-1')
    
    # Feature Engineering
    df = df[['Product Title', 'Bb Category']].copy()
    df['text'] = df['Product Title'] + " " + df['Bb Category']
    df = df[df['text'].str.len() > 5].copy()
    
    # Preprocessing Setup
    stop_words = set(stopwords.words('english'))
    
    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return " ".join(tokens)
    
    df['clean_text'] = df['text'].apply(preprocess)
    df = df.drop_duplicates(subset=['clean_text']).reset_index(drop=True)
    
    # TF-IDF Setup
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    # BERT Setup
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(df['clean_text'].tolist(), show_progress_bar=False)
    
    return df, vectorizer, tfidf_matrix, model, doc_embeddings, preprocess

# Load all core logic and data
df, vectorizer, tfidf_matrix, model, doc_embeddings, preprocess = load_resources()

# --- Search Functions ---
def search_tfidf(query, top_k=5):
    query_clean = preprocess(query)
    query_vec = vectorizer.transform([query_clean])
    similarity = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarity.argsort()[::-1][:top_k]
    
    results = df.iloc[top_indices].copy()
    results['score'] = similarity[top_indices]
    return results

def search_bert(query, top_k=5):
    query_clean = preprocess(query)
    query_embedding = model.encode([query_clean])
    similarity = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarity.argsort()[::-1][:top_k]
    
    results = df.iloc[top_indices].copy()
    results['score'] = similarity[top_indices]
    return results

# --- Advanced Custom CSS and Animations ---
st.markdown("""
<style>
    /* Animated Aurora Background */
    .stApp {
        background: linear-gradient(-45deg, #050505, #0a0e17, #101628, #050505);
        background-size: 400% 400%;
        animation: aurora 15s ease infinite;
        color: #e0e0e0;
    }

    @keyframes aurora {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating Glow Blobs */
    .blob {
        position: fixed;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(0, 242, 254, 0.08) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(100px);
        z-index: -1;
        pointer-events: none;
    }
    .blob-1 { top: -10%; left: -10%; animation: float 25s infinite alternate; }
    .blob-2 { bottom: -10%; right: -10%; animation: float 30s infinite alternate-reverse; }

    @keyframes float {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(150px, 100px) scale(1.2); }
    }

    /* Premium Header Typing Effect */
    .hero-container {
        text-align: center;
        padding: 4rem 1rem;
        animation: fadeIn 2s ease-out;
    }

    .main-title {
        font-size: 4.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #ffffff 0%, #a0a0a0 50%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
        margin-bottom: 0px !important;
        line-height: 1.1;
    }

    .sub-title {
        font-size: 1.3rem;
        color: #4facfe;
        font-weight: 500;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 1rem;
        opacity: 0.8;
    }

    /* Search Container Glassmorphism */
    .search-section {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin-bottom: 3rem;
        animation: fadeInUp 1s cubic-bezier(0.23, 1, 0.32, 1);
    }

    /* Premium Card Design */
    .result-card {
        background: rgba(15, 15, 25, 0.4);
        backdrop-filter: blur(12px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        animation: cardAppear 0.8s cubic-bezier(0.23, 1, 0.32, 1) forwards;
        opacity: 0;
        transform: translateY(30px);
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            rgba(79, 172, 254, 0.05),
            transparent
        );
        transition: 0.5s;
    }

    .result-card:hover {
        transform: translateY(-10px) scale(1.02) rotateX(2deg);
        border-color: rgba(79, 172, 254, 0.3);
        background: rgba(20, 20, 35, 0.6);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(79, 172, 254, 0.1);
    }

    .result-card:hover::before {
        left: 100%;
    }

    .card-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #4facfe;
        margin-bottom: 0.5rem;
        display: block;
    }

    .result-title {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        line-height: 1.3;
    }

    .score-badge {
        display: inline-block;
        background: rgba(79, 172, 254, 0.1);
        color: #4facfe;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(79, 172, 254, 0.2);
    }

    /* Animations */
    @keyframes cardAppear {
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Input & Button Overrides */
    .stTextInput input {
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(0, 0, 0, 0.2) !important;
        padding: 15px 20px !important;
        font-size: 1.1rem !important;
        height: auto !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        height: 56px !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 15px !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #050505; }
    ::-webkit-scrollbar-thumb { background: #1a1a1a; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #4facfe; }
</style>

<!-- Background Elements -->
<div class="blob blob-1"></div>
<div class="blob blob-2"></div>
""", unsafe_allow_html=True)

# --- Streamlit UI Layout ---
st.markdown("""
<div class="hero-container">
    <h1 class="main-title">INTELLIGENT<br>SEARCH</h1>
    <p class="sub-title">Neural Search Engine v2.0</p>
</div>
""", unsafe_allow_html=True)

# Main Search Panel
with st.container():
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("", placeholder="Describe what you're looking for...", label_visibility="collapsed")
    with col2:
        model_choice = st.selectbox("", ["BERT Semantic Search", "TF-IDF Keyword Search"], label_visibility="collapsed")
    
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        search_clicked = st.button("Initialize Search")
    st.markdown('</div>', unsafe_allow_html=True)

if search_clicked:
    if not user_query.strip():
        st.warning("Please enter a query to search.")
    else:
        with st.spinner("🧠 Analyzing semantics and crawling index..."):
            if "TF-IDF" in model_choice:
                results = search_tfidf(user_query)
            else:
                results = search_bert(user_query)
            
        # Results Header
        st.markdown(f"""
        <div style="margin-bottom: 2rem; opacity: 0.7;">
            <span style="font-size: 0.9rem; letter-spacing: 1px;">SHOWING RESULTS FOR:</span>
            <h3 style="color: #4facfe; margin-top: 0;">"{user_query}"</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if results.empty or results['score'].max() == 0:
            st.info("No neural matches found. Try broadening your terms.")
        else:
            # Display results using Ultra Premium Cards
            for i, (idx, row) in enumerate(results.iterrows()):
                score_val = row['score']
                st.markdown(f"""
                <div class="result-card" style="animation-delay: {i*0.15}s">
                    <span class="card-label">{row['Bb Category']}</span>
                    <div class="result-title">{row['Product Title']}</div>
                    <div class="score-badge">Neural Match: {score_val:.2%}</div>
                </div>
                """, unsafe_allow_html=True)

# Advanced Footer
st.markdown("""
<div style="margin-top: 5rem; border-top: 1px solid rgba(255,255,255,0.05); padding: 3rem 0; text-align: center;">
    <div style="font-size: 0.7rem; color: #4facfe; letter-spacing: 3px; font-weight: 700; margin-bottom: 1rem;">
        SYSTEM STATUS: ONLINE
    </div>
    <div style="font-size: 0.8rem; opacity: 0.4;">
        Utilizing Transformers & Vector Embeddings for Semantic Precision
    </div>
</div>
""", unsafe_allow_html=True)
