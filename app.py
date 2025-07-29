import streamlit as st
import joblib
import json
import os
import random

MODEL_PATH = "disease_predictor_model.joblib"
STATS_FILE = "stats.json"

# Load model once with caching
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# Load disease stats or create empty file
def load_stats():
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, "w") as f:
            json.dump({}, f)
        return {}
    with open(STATS_FILE, "r") as f:
        return json.load(f)

# Save updated stats
def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

# Increment disease count
def update_stats(stats, disease):
    if disease in stats:
        stats[disease] += 1
    else:
        stats[disease] = 1
    save_stats(stats)

model = load_model()
stats = load_stats()

# Page config
st.set_page_config(
    page_title="AI Disease Predictor", 
    page_icon="🏥", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Modern, clean CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    
    /* Header section */
    .header-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #718096;
        font-weight: 400;
        line-height: 1.5;
    }
    
    /* Input section */
    .input-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        transition: all 0.3s ease !important;
        background: #ffffff !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Result section */
    .result-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #48bb78;
    }
    
    .result-title {
        font-size: 1.2rem;
        color: #718096;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .result-disease {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .result-note {
        font-size: 0.9rem;
        color: #a0aec0;
        font-style: italic;
    }
    
    /* Stats section */
    .stats-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    }
    
    .stats-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stat-item:last-child {
        border-bottom: none;
    }
    
    .disease-name {
        font-weight: 600;
        color: #4a5568;
        flex: 1;
    }
    
    .progress-container {
        flex: 2;
        margin: 0 1rem;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.8s ease;
    }
    
    .stat-count {
        font-weight: 600;
        color: #667eea;
        min-width: 40px;
        text-align: right;
    }
    
    /* Info message styling */
    .stInfo {
        background: rgba(235, 248, 255, 0.95) !important;
        border-radius: 12px !important;
        border-left: 4px solid #0ea5e9 !important;
    }
    
    .stWarning {
        background: rgba(255, 251, 235, 0.95) !important;
        border-radius: 12px !important;
        border-left: 4px solid #f59e0b !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .input-container, .stats-container, .result-container, .header-container {
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .stat-item {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.5rem;
        }
        
        .progress-container {
            width: 100%;
            margin: 0;
        }
        
        .stat-count {
            align-self: flex-end;
        }
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="header-container">
    <div class="main-title">🏥 AI Disease Predictor</div>
    <div class="subtitle">
        Get instant predictions based on your symptoms using advanced machine learning.<br>
        Simply enter your symptoms separated by commas (e.g., "fever, cough, headache")
    </div>
</div>
""", unsafe_allow_html=True)

# Input section
with st.container():
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    user_input = st.text_input(
        "Enter your symptoms",
        placeholder="e.g., fever, cough, sore throat, headache",
        help="Separate multiple symptoms with commas for better accuracy"
    )
    
    predict_button = st.button("🔍 Analyze Symptoms", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if predict_button:
    if user_input.strip():
        predicted_label = model.predict([user_input])[0]
        
        # Show success animation
        st.balloons()
        
        # Display result
        st.markdown(f"""
        <div class="result-container">
            <div class="result-title">Analysis Complete</div>
            <div class="result-disease">{predicted_label}</div>
            <div class="result-note">
                ⚠️ This is an AI prediction for informational purposes only.<br>
                Please consult a healthcare professional for proper diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Update stats
        update_stats(stats, predicted_label)
        stats = load_stats()  # reload updated stats
    else:
        st.warning("Please enter at least one symptom to get a prediction.")

# Statistics section
st.markdown('<div class="stats-container">', unsafe_allow_html=True)
st.markdown('<div class="stats-title">📊 Recent Predictions</div>', unsafe_allow_html=True)

if stats:
    max_count = max(stats.values())
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    
    for disease, count in sorted_stats.items():
        percentage = int((count / max_count) * 100)
        
        st.markdown(f"""
        <div class="stat-item">
            <div class="disease-name">{disease}</div>
            <div class="progress-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {percentage}%;"></div>
                </div>
            </div>
            <div class="stat-count">{count}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("💡 No predictions yet. Be the first to try the disease predictor!")

st.markdown('</div>', unsafe_allow_html=True)
