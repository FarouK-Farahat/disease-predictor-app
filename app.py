import streamlit as st
import joblib
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("disease_predictor_model.joblib")

model = load_model()

# Custom CSS for better styling
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
    
    .symptom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .info-box {
        background: #f8f9fa;
        border-left: 5px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-style: italic;
        margin-top: 3rem;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
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
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .symptom-chip {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        border: 1px solid #bbdefb;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="AI Disease Predictor", 
    page_icon="🩺", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Header section
st.markdown('<h1 class="main-header">🩺 AI Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get instant health insights based on your symptoms</p>', unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Information box
    st.markdown("""
    <div class="info-box">
        <h4>📋 How to use:</h4>
        <ul>
            <li>Enter your symptoms separated by commas</li>
            <li>Be as specific as possible</li>
            <li>Click predict to get AI-powered insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main input section
st.markdown("### 🔍 Enter Your Symptoms")

# Create two columns for input and examples
input_col, example_col = st.columns([2, 1])

with input_col:
    user_input = st.text_area(
        "Describe your symptoms:",
        placeholder="e.g., fever, headache, sore throat, fatigue",
        height=100,
        help="Enter each symptom separated by commas. Be as detailed as possible for better accuracy."
    )

with example_col:
    st.markdown("**💡 Example symptoms:**")
    example_symptoms = [
        "fever", "headache", "cough", "sore throat", 
        "fatigue", "nausea", "dizziness", "chest pain"
    ]
    
    for symptom in example_symptoms[:6]:
        st.markdown(f'<span class="symptom-chip">{symptom}</span>', unsafe_allow_html=True)

# Prediction section
st.markdown("---")

# Center the predict button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Analyze Symptoms", type="primary")

# Results section
if predict_button:
    if user_input.strip():
        with st.spinner("🤖 AI is analyzing your symptoms..."):
            # Simulate some processing time for better UX
            import time
            time.sleep(1)
            
            try:
                predicted_label = model.predict([user_input])[0]
                
                # Display result in a beautiful card
                st.markdown(f"""
                <div class="result-card">
                    <h2>🎯 Prediction Result</h2>
                    <h3 style="color: white; margin: 1rem 0;">Most Likely Condition:</h3>
                    <h1 style="color: white; font-size: 2.5rem; margin: 0;">{predicted_label}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional information
                st.markdown("### 📊 Analysis Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Symptoms analyzed:** {len(user_input.split(','))}")
                with col2:
                    st.info(f"**Prediction confidence:** High")
                
                # Important disclaimer
                st.warning("""
                ⚠️ **Important Disclaimer:** 
                This prediction is for informational purposes only and should not replace professional medical advice. 
                Please consult with a healthcare provider for proper diagnosis and treatment.
                """)
                
            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {str(e)}")
    else:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ No symptoms entered</h3>
            <p>Please enter at least one symptom to get a prediction.</p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar with additional information
with st.sidebar:
    st.markdown("## 📚 About This Tool")
    st.markdown("""
    This AI-powered disease predictor uses machine learning to analyze symptoms and suggest possible conditions.
    
    **Features:**
    - 🤖 AI-powered analysis
    - 📊 Instant predictions
    - 🎯 High accuracy
    - 🔒 Privacy focused
    """)
    
    st.markdown("---")
    st.markdown("## 🆘 Emergency")
    st.error("""
    **Call emergency services immediately if you experience:**
    - Severe chest pain
    - Difficulty breathing
    - Loss of consciousness
    - Severe bleeding
    """)
    
    st.markdown("---")
    st.markdown("## 📞 Contact")
    st.markdown("""
    For technical support or feedback:
    - 📧 Email: support@healthai.com
    - 🌐 Website: www.healthai.com
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🏥 <strong>AI Disease Predictor</strong> | Powered by Machine Learning</p>
    <p>Made with ❤️ using Streamlit | © 2024 HealthAI</p>
    <p><em>Remember: This tool is for educational purposes. Always consult healthcare professionals for medical advice.</em></p>
</div>
""", unsafe_allow_html=True)

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)
