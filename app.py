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
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="centered")

# Custom CSS for stunning UI
st.markdown(
    """
    <style>
    /* Background gradient */
    .main {
        background: linear-gradient(135deg, #f0f5ff 0%, #dbeafe 100%);
        min-height: 100vh;
        padding: 3rem 2rem 5rem 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e293b;
    }
    h1 {
        text-align: center;
        font-size: 3.8rem;
        font-weight: 800;
        color: #1e40af;
        margin-bottom: 0.15rem;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.4);
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #334155;
        margin-bottom: 3rem;
        font-weight: 500;
    }
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 18px !important;
        border: 3px solid #3b82f6 !important;
        padding: 1rem 1.2rem !important;
        font-size: 1.3rem !important;
        box-shadow: 0 0 15px #3b82f6aa;
        transition: all 0.3s ease;
        font-weight: 600;
        letter-spacing: 0.04em;
        background: #fefefe;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 20px #2563ebcc;
        outline: none !important;
        background: #fff;
    }
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        border: none;
        border-radius: 20px;
        color: white;
        font-size: 1.4rem;
        padding: 1rem 2.5rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.5);
        transition: background 0.4s, box-shadow 0.3s;
        width: 100%;
        letter-spacing: 0.05em;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1e40af, #2563eb);
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.7);
    }
    /* Result box */
    .result-box {
        background: white;
        border-radius: 25px;
        padding: 2rem 3rem;
        box-shadow: 0 0 30px #3b82f6bb;
        margin-top: 2rem;
        text-align: center;
        font-size: 2.1rem;
        font-weight: 900;
        color: #1e40af;
        letter-spacing: 0.05em;
        user-select: none;
        transition: transform 0.3s ease;
    }
    .result-box:hover {
        transform: scale(1.05);
    }
    /* Stats container */
    .stats-container {
        background: rgba(255,255,255,0.95);
        border-radius: 25px;
        padding: 2rem 3rem;
        box-shadow: 0 0 40px #2563eb55;
        margin-top: 4rem;
    }
    .stats-title {
        font-size: 2.5rem;
        font-weight: 900;
        color: #1e3a8a;
        margin-bottom: 2rem;
        text-align: center;
        letter-spacing: 0.1em;
        text-shadow: 0 0 8px #2563ebaa;
    }
    .stat-item {
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: #1e40af;
        font-weight: 700;
        letter-spacing: 0.02em;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    /* Progress bar container */
    .progress-bar {
        flex-grow: 1;
        height: 20px;
        background: #cbd5e1;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: inset 0 0 8px #94a3b8;
    }
    /* Progress bar fill */
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        border-radius: 20px 0 0 20px;
        transition: width 1s ease;
    }
    /* Disease name */
    .disease-name {
        min-width: 150px;
        text-align: right;
        color: #1e40af;
        font-weight: 800;
        user-select: text;
    }
    /* Responsive tweaks */
    @media (max-width: 480px) {
        .result-box {
            font-size: 1.6rem;
            padding: 1.5rem 2rem;
        }
        .stats-title {
            font-size: 2rem;
        }
        .stat-item {
            font-size: 1.1rem;
            flex-direction: column;
            gap: 0.3rem;
        }
        .disease-name {
            text-align: center;
            min-width: auto;
            user-select: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main">', unsafe_allow_html=True)

# Header Section
st.markdown('''
    <div style="display: flex; flex-direction: column; align-items: center; margin-bottom: 2.5rem;">
        <div style="background: linear-gradient(90deg, #2563eb 60%, #3b82f6 100%); border-radius: 30px; box-shadow: 0 8px 32px #3b82f655; padding: 2.5rem 2rem 2rem 2rem; width: 100%; max-width: 600px;">
            <h1 style="color: #fff; font-size: 3rem; font-weight: 900; margin-bottom: 0.5rem; text-align: center; letter-spacing: 2px; text-shadow: 0 0 12px #1e40af99;">🩺 Symptom-Based Disease Predictor</h1>
            <p class="subtitle" style="color: #e0e7ef; font-size: 1.2rem; text-align: center; margin-bottom: 0;">Enter your symptoms below, separated by commas.<br><em>Example: fever, cough, sore throat</em></p>
        </div>
    </div>
''', unsafe_allow_html=True)

# Input Card
st.markdown('''
    <div style="display: flex; justify-content: center;">
        <div style="background: #fff; border-radius: 25px; box-shadow: 0 4px 24px #2563eb33; padding: 2.5rem 2rem 2rem 2rem; width: 100%; max-width: 500px; margin-bottom: 2rem;">
''', unsafe_allow_html=True)

user_input = st.text_input("Symptoms", "", placeholder="e.g. fever, cough, sore throat")

predicted_label = None
if st.button("🔍 Predict Disease"):
    if user_input.strip():
        predicted_label = model.predict([user_input])[0]
        # Show confetti celebration 🎉
        st.balloons()
        st.markdown(f'<div class="result-box" style="margin-top: 1.5rem;">Most Likely Disease: <span style="color: #16a34a;">{predicted_label}</span></div>', unsafe_allow_html=True)
        update_stats(stats, predicted_label)
        stats = load_stats()  # reload updated stats
    else:
        st.warning("Please enter at least one symptom.")

st.markdown('</div></div>', unsafe_allow_html=True)

# Stats Section
st.markdown('<div class="stats-container">', unsafe_allow_html=True)
st.markdown('<h2 class="stats-title">📊 Disease Reports So Far</h2>', unsafe_allow_html=True)

if stats:
    max_count = max(stats.values())
    sorted_stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
    for disease, count in sorted_stats.items():
        bar_length = int((count / max_count) * 100)
        st.markdown(
            f'''
            <div class="stat-item" style="background: #f1f5fb; border-radius: 18px; padding: 0.7rem 1.2rem; margin-bottom: 1.2rem; box-shadow: 0 2px 8px #2563eb11;">
                <div class="disease-name" style="font-size: 1.1rem; color: #1e40af; font-weight: 800; min-width: 120px;">🦠 {disease}</div>
                <div class="progress-bar" aria-label="{disease} reports: {count}" style="height: 18px; background: #e0e7ef;">
                    <div class="progress-fill" style="width: {bar_length}%; background: linear-gradient(90deg, #16a34a, #3b82f6); height: 100%; border-radius: 20px 0 0 20px;"></div>
                </div>
                <div style="min-width: 45px; text-align: right; font-weight: 700; color: #16a34a; font-size: 1.1rem;">{count}</div>
            </div>
            ''',
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
