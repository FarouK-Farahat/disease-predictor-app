import streamlit as st
import pandas as pd
import joblib
import warnings
import re

# Suppress warnings for a cleaner interface
warnings.filterwarnings("ignore", category=UserWarning)

# --- Load Model and Data ---
@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline."""
    try:
        pipeline = joblib.load('disease_prediction_pipeline.joblib')
        return pipeline
    except FileNotFoundError:
        return None

@st.cache_data
def load_and_analyze_data():
    """Load, clean, and analyze the dataset for the analytics pages."""
    try:
        # Load the raw data
        df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

        # --- Data Cleaning (mirrors the training script) ---
        df['Outcome Variable'] = df['Outcome Variable'].replace('Positiveread', 'Positive')
        df['Disease'] = df['Disease'].apply(lambda x: re.sub(r'\s*\(.*\)\s*', '', str(x)).strip())
        df['Disease'] = df['Disease'].apply(lambda x: re.sub(r'\.{3}$', '', str(x)).strip())
        df.drop_duplicates(inplace=True)
        
        # --- Analytics DataFrames ---
        disease_stats = df['Disease'].value_counts().reset_index()
        disease_stats.columns = ['Disease', 'Cases']
        
        return df, disease_stats
    except FileNotFoundError:
        return None, None

# Load the necessary components
pipeline = load_model()
df, disease_stats = load_and_analyze_data()

# --- UI and Page Configuration ---
st.set_page_config(
    page_title="AI Disease Predictor", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS from your design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .team-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- App Logic Check ---
if pipeline is None or df is None:
    st.error("🚨 Critical files are missing! Please ensure both `disease_prediction_pipeline.joblib` and `Disease_symptom_and_patient_profile_dataset.csv` are in the same directory as the app.")
    st.stop()

# --- Navigation ---
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Choose a section:", 
                           ["🏠 Home", "📊 Analytics", "📋 Dataset", "👥 Our Team"])

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🩺 AI Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get instant health insights from patient data</p>', unsafe_allow_html=True)

    # Key metrics from the loaded dataset
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"> <div style="font-size: 2.5rem; font-weight: bold;">{len(df)}</div> <div>Unique Records</div> </div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"> <div style="font-size: 2.5rem; font-weight: bold;">{len(disease_stats)}</div> <div>Diseases Covered</div> </div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"> <div style="font-size: 2.5rem; font-weight: bold;">8</div> <div>Predictive Features</div> </div>', unsafe_allow_html=True)

    st.markdown("---")

    # --- User Input Form ---
    with st.form("prediction_form"):
        st.header("🔍 Enter Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35, step=1)
            gender = st.selectbox("Gender", df['Gender'].unique())
            blood_pressure = st.selectbox("Blood Pressure", df['Blood Pressure'].unique())
            cholesterol_level = st.selectbox("Cholesterol Level", df['Cholesterol Level'].unique())
        with col2:
            fever = st.selectbox("Fever", df['Fever'].unique())
            cough = st.selectbox("Cough", df['Cough'].unique())
            fatigue = st.selectbox("Fatigue", df['Fatigue'].unique())
            difficulty_breathing = st.selectbox("Difficulty Breathing", df['Difficulty Breathing'].unique())
        
        submit_button = st.form_submit_button(label='🩺 Predict Disease')

    # --- Prediction Logic ---
    if submit_button:
        input_data = pd.DataFrame({
            'Fever': [fever], 'Cough': [cough], 'Fatigue': [fatigue], 
            'Difficulty Breathing': [difficulty_breathing], 'Age': [age], 'Gender': [gender],
            'Blood Pressure': [blood_pressure], 'Cholesterol Level': [cholesterol_level]
        })
        with st.spinner("🤖 AI is analyzing the data..."):
            prediction = pipeline.predict(input_data)
            prediction_proba = pipeline.predict_proba(input_data)
            confidence_score = prediction_proba.max() * 100
        
        st.markdown(f"""
        <div class="result-card">
            <h2>🎯 Prediction Result</h2>
            <h3 style="color: white; margin: 1rem 0;">Predicted Condition:</h3>
            <h1 style="color: white; font-size: 2.5rem; margin: 0;">{prediction}</h1>
            <h4 style="color: white; margin-top: 1rem;">Confidence: {confidence_score:.2f}%</h4>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("⚠️ Disclaimer: This tool is for informational purposes only. It is not a substitute for professional medical advice.", icon="ℹ️")


elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Dataset Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visual insights from the underlying patient data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🦠 Disease Distribution")
        st.bar_chart(disease_stats.set_index('Disease')['Cases'])
    with col2:
        st.subheader("🩸 Blood Pressure Distribution")
        bp_dist = df['Blood Pressure'].value_counts()
        st.bar_chart(bp_dist)
        
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("🧬 Gender Distribution")
        gender_dist = df['Gender'].value_counts()
        st.bar_chart(gender_dist)
    with col4:
        st.subheader("🧪 Cholesterol Level Distribution")
        chol_dist = df['Cholesterol Level'].value_counts()
        st.bar_chart(chol_dist)
        
    st.subheader("🧑 Age Distribution of Patients")
    st.bar_chart(df['Age'].value_counts())


elif page == "📋 Dataset":
    st.markdown('<h1 class="main-header">📋 Dataset Overview</h1>', unsafe_allow_html=True)
    st.info("This is a preview of the cleaned dataset used to train the prediction model.")
    st.dataframe(df.head(50))

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Cleaned Dataset (CSV)",
        data=csv_data,
        file_name="cleaned_disease_symptom_dataset.csv",
        mime="text/csv"
    )

elif page == "👥 Our Team":
    st.markdown('<h1 class="main-header">👥 Meet The Team</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The minds behind this AI-powered health tool</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="team-card"><h3>👨‍💻 Dr. Alex Johnson</h3><p><strong>Lead Data Scientist</strong><br>PhD in Machine Learning</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="team-card"><h3>👩‍⚕️ Dr. Sarah Chen</h3><p><strong>Medical Advisor</strong><br>MD, Internal Medicine</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="team-card"><h3>🛠️ Mike Rodriguez</h3><p><strong>Software Engineer</strong><br>MS in Computer Science</p></div>', unsafe_allow_html=True)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.error("""
**Medical Emergency?**
If you are experiencing a medical emergency, please call your local emergency number immediately.
""")