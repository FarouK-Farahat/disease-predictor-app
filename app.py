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
    page_icon="🩺", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with modern design elements
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-1d391kg {padding: 0;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    
    /* Main background with animated gradient */
    .main {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
        padding: 2rem 1rem 3rem 1rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a202c;
        position: relative;
        overflow-x: hidden;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Floating particles background */
    .main::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(255,255,255,0.1) 2px, transparent 2px),
            radial-gradient(circle at 75% 75%, rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 100px 100px, 50px 50px;
        animation: float 20s linear infinite;
        pointer-events: none;
        z-index: 1;
    }
    
    @keyframes float {
        0% { transform: translateY(0px) translateX(0px); }
        33% { transform: translateY(-10px) translateX(5px); }
        66% { transform: translateY(5px) translateX(-5px); }
        100% { transform: translateY(0px) translateX(0px); }
    }
    
    /* Content container with glassmorphism */
    .content-container {
        position: relative;
        z-index: 2;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 30px;
        padding: 3rem 2.5rem;
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        max-width: 800px;
        margin: 0 auto;
        animation: slideInUp 0.8s ease-out;
    }
    
    @keyframes slideInUp {
        from { 
            opacity: 0; 
            transform: translateY(50px); 
        }
        to { 
            opacity: 1; 
            transform: translateY(0); 
        }
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        line-height: 1.1;
        text-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        0% { filter: brightness(1) drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
        100% { filter: brightness(1.1) drop-shadow(0 0 20px rgba(102, 126, 234, 0.6)); }
    }
    
    .subtitle {
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        line-height: 1.6;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .highlight {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.2rem 0.6rem;
        border-radius: 10px;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
    
    /* Input section styling */
    .input-section {
        margin: 3rem 0;
        position: relative;
    }
    
    .input-label {
        display: block;
        font-size: 1.2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 20px !important;
        padding: 1.2rem 1.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #2d3748 !important;
        box-shadow: 
            0 10px 25px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.6) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.2),
            0 0 0 4px rgba(102, 126, 234, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
        outline: none !important;
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #a0aec0 !important;
        font-style: italic;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 20px !important;
        color: white !important;
        font-size: 1.2rem !important;
        padding: 1rem 2rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        position: relative !important;
        overflow: hidden !important;
        box-shadow: 
            0 10px 25px rgba(102, 126, 234, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 
            0 15px 35px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4c93 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Result box enhancement */
    .result-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
        margin: 2rem 0;
        text-align: center;
        animation: resultAppear 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes resultAppear {
        0% {
            opacity: 0;
            transform: scale(0.8) translateY(20px);
        }
        100% {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }
    
    .result-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        animation: shimmer 2s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .result-title {
        font-size: 1.4rem;
        color: #4a5568;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .result-disease {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        line-height: 1.2;
    }
    
    /* Stats container enhancement */
    .stats-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 25px;
        padding: 2.5rem;
        box-shadow: 
            0 25px 50px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
        margin-top: 3rem;
        animation: slideInUp 0.8s ease-out 0.3s both;
    }
    
    .stats-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.2rem;
        padding: 0.8rem;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        animation: statItemAppear 0.5s ease-out both;
    }
    
    .stat-item:nth-child(2) { animation-delay: 0.1s; }
    .stat-item:nth-child(3) { animation-delay: 0.2s; }
    .stat-item:nth-child(4) { animation-delay: 0.3s; }
    .stat-item:nth-child(5) { animation-delay: 0.4s; }
    
    @keyframes statItemAppear {
        0% {
            opacity: 0;
            transform: translateX(-20px);
        }
        100% {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stat-item:hover {
        transform: translateX(5px);
        background: rgba(255, 255, 255, 0.6);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .disease-name {
        min-width: 140px;
        font-weight: 700;
        color: #2d3748;
        font-size: 1rem;
    }
    
    .progress-bar {
        flex-grow: 1;
        height: 12px;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        animation: progressShine 2s linear infinite;
    }
    
    @keyframes progressShine {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .count-value {
        min-width: 40px;
        text-align: center;
        font-weight: 800;
        color: #2d3748;
        font-size: 1rem;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 8px;
        padding: 0.3rem 0.6rem;
    }
    
    /* Info message styling */
    .stInfo {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Warning message styling */
    .stWarning {
        background: rgba(255, 193, 7, 0.9) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 193, 7, 0.3) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .content-container {
            padding: 2rem 1.5rem;
            margin: 1rem;
        }
        
        .main-title {
            font-size: 2.8rem;
        }
        
        .subtitle {
            font-size: 1.1rem;
        }
        
        .result-disease {
            font-size: 2rem;
        }
        
        .stats-title {
            font-size: 1.8rem;
        }
        
        .stat-item {
            flex-direction: column;
            gap: 0.5rem;
            text-align: center;
        }
        
        .disease-name {
            min-width: auto;
        }
        
        .progress-bar {
            width: 100%;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 2.2rem;
        }
        
        .result-disease {
            font-size: 1.6rem;
        }
        
        .stats-title {
            font-size: 1.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main content container
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# Header section
st.markdown(
    '''
    <div class="header-container">
        <h1 class="main-title">🩺 AI Disease Predictor</h1>
        <p class="subtitle">
            Advanced symptom analysis powered by machine learning<br>
            Enter your symptoms below, separated by commas<br>
            <span class="highlight">Example: fever, cough, sore throat, headache</span>
        </p>
    </div>
    ''', 
    unsafe_allow_html=True
)

# Input section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown('<label class="input-label">🔍 Describe Your Symptoms</label>', unsafe_allow_html=True)
user_input = st.text_input(
    "Symptoms", 
    "",
    placeholder="Enter symptoms separated by commas (e.g., fever, cough, headache)...",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

predicted_label = None
if st.button("🚀 Analyze Symptoms"):
    if user_input.strip():
        predicted_label = model.predict([user_input])[0]
        # Show celebration
        st.balloons()
        st.markdown(
            f'''
            <div class="result-box">
                <div class="result-title">🎯 Predicted Condition</div>
                <div class="result-disease">{predicted_label}</div>
            </div>
            ''', 
            unsafe_allow_html=True
        )
        update_stats(stats, predicted_label)
        stats = load_stats()  # reload updated stats
    else:
        st.warning("⚠️ Please enter at least one symptom to get a prediction.")

# Statistics section
st.markdown('<div class="stats-container">', unsafe_allow_html=True)
st.markdown('<h2 class="stats-title">📊 Disease Prediction Analytics</h2>', unsafe_allow_html=True)

if stats:
    max_count = max(stats.values())
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    total_predictions = sum(stats.values())
    
    # Add total predictions info
    st.markdown(
        f'''
        <div style="text-align: center; margin-bottom: 2rem; padding: 1rem; background: rgba(255,255,255,0.3); border-radius: 15px; backdrop-filter: blur(10px);">
            <div style="font-size: 1.1rem; color: #4a5568; font-weight: 600;">Total Predictions Made</div>
            <div style="font-size: 2rem; font-weight: 800; color: #2d3748; margin-top: 0.5rem;">{total_predictions}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    for i, (disease, count) in enumerate(sorted_stats.items()):
        bar_length = int((count / max_count) * 100) if max_count > 0 else 0
        percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
        
        # Add ranking emoji for top 3
        ranking_emoji = ""
        if i == 0:
            ranking_emoji = "🥇 "
        elif i == 1:
            ranking_emoji = "🥈 "
        elif i == 2:
            ranking_emoji = "🥉 "
        
        st.markdown(
            f'''
            <div class="stat-item">
                <div class="disease-name">{ranking_emoji}{disease}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {bar_length}%;"></div>
                </div>
                <div class="count-value">{count}</div>
                <div style="min-width: 60px; text-align: right; font-size: 0.9rem; color: #6b7280; font-weight: 500;">
                    {percentage:.1f}%
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
else:
    st.info("📋 No predictions yet. Be the first to try our AI disease predictor!")

st.markdown('</div>', unsafe_allow_html=True)  # Close stats container

# Footer
st.markdown(
    '''
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.7); font-size: 0.9rem;">
        <p style="margin-bottom: 0.5rem;">⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only.</p>
        <p>Always consult with healthcare professionals for medical advice.</p>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)  # Close content container
