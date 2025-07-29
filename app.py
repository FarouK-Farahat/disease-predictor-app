import streamlit as st
import joblib
import json
import os

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
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        min-height: 100vh;
        padding: 3rem 2rem 5rem 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1f2937;
    }
    h1 {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.1rem;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #334155;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 15px !important;
        border: 2px solid #3b82f6 !important;
        padding: 0.8rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 0 8px #3b82f6aa;
        transition: 0.3s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 12px #2563ebcc;
        outline: none !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        border: none;
        border-radius: 15px;
        color: white;
        font-size: 1.2rem;
        padding: 0.7rem 1.7rem;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(59,130,246,0.4);
        transition: background 0.3s;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1e40af, #2563eb);
        cursor: pointer;
    }
    .result-box {
        background: white;
        border-radius: 15px;
        padding: 1.5rem 2rem;
        box-shadow: 0 0 12px #3b82f6aa;
        margin-top: 1.8rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563eb;
    }
    .stats-container {
        background: rgba(255,255,255,0.85);
        border-radius: 15px;
        padding: 1.5rem 2rem;
        box-shadow: 0 0 15px #2563eb55;
        margin-top: 3rem;
    }
    .stats-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 1rem;
        text-align: center;
    }
    .stat-item {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.markdown('<h1>🩺 Symptom-Based Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your symptoms below, separated by commas.<br><em>Example: fever, cough, sore throat</em></p>', unsafe_allow_html=True)

# Input box
user_input = st.text_input("Symptoms", "")

predicted_label = None
if st.button("🔍 Predict Disease"):
    if user_input.strip():
        predicted_label = model.predict([user_input])[0]
        st.markdown(f'<div class="result-box">Most Likely Disease: <span>{predicted_label}</span></div>', unsafe_allow_html=True)
        update_stats(stats, predicted_label)
        stats = load_stats()  # reload updated stats
    else:
        st.warning("Please enter at least one symptom.")

# Show stats section
st.markdown('<div class="stats-container">', unsafe_allow_html=True)
st.markdown('<h2 class="stats-title">📊 Disease Reports So Far</h2>', unsafe_allow_html=True)

if stats:
    max_count = max(stats.values())
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    for disease, count in sorted_stats.items():
        bar_length = int((count / max_count) * 100)
        bar_color = "#3b82f6"
        st.markdown(
            f"""
            <div class="stat-item">
                <strong>{disease}</strong>: {count} reports
                <div style="background:#cbd5e1; border-radius:8px; margin-top:4px; height:16px; width:100%; max-width:400px;">
                    <div style="background:{bar_color}; width:{bar_length}%; height:16px; border-radius:8px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("No disease reports yet. Be the first to predict!")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Remove default Streamlit footer ("Made with Streamlit")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
