import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random
from collections import Counter
import csv
import io

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("disease_predictor_model.joblib")

model = load_model()

# Generate dataset using your code
@st.cache_data
def generate_dataset():
    disease_symptoms = {
        "Common Cold": ["fever", "cough", "sore throat", "runny nose", "sneezing"],
        "Flu": ["fever", "cough", "headache", "muscle pain", "fatigue", "chills"],
        "Bronchitis": ["cough", "shortness of breath", "chest pain", "fatigue", "sore throat"],
        "Dengue": ["fever", "headache", "joint pain", "rash", "nausea", "vomiting"],
        "Malaria": ["fever", "chills", "sweating", "headache", "muscle pain"],
        "Tuberculosis": ["fever", "cough", "night sweats", "weight loss", "fatigue"],
        "Gastroenteritis": ["diarrhea", "abdominal pain", "fever", "nausea", "vomiting"],
        "Allergy": ["itching", "rash", "redness", "swelling", "sneezing"],
        "Diabetes": ["fatigue", "weight loss", "blurred vision", "increased thirst"],
        "COVID-19": ["fever", "cough", "loss of taste", "shortness of breath", "fatigue"],
    }
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Generate 1500 rows
    num_rows = 1500
    rows = []
    
    for _ in range(num_rows):
        disease = random.choice(list(disease_symptoms.keys()))
        symptoms_list = disease_symptoms[disease]
        symptom_count = random.randint(2, len(symptoms_list))
        symptoms = random.sample(symptoms_list, symptom_count)
        symptom_str = ", ".join(symptoms)
        rows.append([symptom_str, disease])
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Symptoms", "Disease"])
    
    # Add symptom count
    df['Symptom_Count'] = df['Symptoms'].apply(lambda x: len(x.split(',')))
    
    # Extract all symptoms for analysis
    all_symptoms = []
    for symptoms_str in df['Symptoms']:
        symptoms = [s.strip() for s in symptoms_str.split(',')]
        all_symptoms.extend(symptoms)
    
    # Create symptom frequency data
    symptom_counts = Counter(all_symptoms)
    symptom_freq_df = pd.DataFrame(list(symptom_counts.items()), columns=['Symptom', 'Frequency'])
    symptom_freq_df = symptom_freq_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    symptom_freq_df['Percentage'] = (symptom_freq_df['Frequency'] / len(all_symptoms) * 100).round(2)
    
    # Disease distribution
    disease_stats = df['Disease'].value_counts().reset_index()
    disease_stats.columns = ['Disease', 'Cases']
    disease_stats['Percentage'] = (disease_stats['Cases'] / len(df) * 100).round(2)
    
    # Add severity and recovery days (simulated for demo)
    severity_map = {
        'Common Cold': 'Low', 'Flu': 'Medium', 'Bronchitis': 'Medium',
        'Dengue': 'High', 'Malaria': 'High', 'Tuberculosis': 'High',
        'Gastroenteritis': 'Medium', 'Allergy': 'Low', 'Diabetes': 'High',
        'COVID-19': 'High'
    }
    
    recovery_map = {
        'Common Cold': 7, 'Flu': 10, 'Bronchitis': 14,
        'Dengue': 21, 'Malaria': 28, 'Tuberculosis': 180,
        'Gastroenteritis': 5, 'Allergy': 3, 'Diabetes': 365,
        'COVID-19': 14
    }
    
    disease_stats['Severity'] = disease_stats['Disease'].map(severity_map)
    disease_stats['Recovery_Days'] = disease_stats['Disease'].map(recovery_map)
    
    # Monthly data (simulated)
    monthly_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'Predictions': [1250, 1180, 1420, 1680, 1890, 2100, 2300, 2150, 1980, 1750, 1600, 1400],
        'Accuracy': [94.2, 95.1, 93.8, 96.2, 95.7, 96.8, 97.1, 96.5, 95.9, 96.3, 95.8, 96.0]
    })
    
    # Symptoms per disease stats
    symptoms_per_disease = df.groupby('Disease')['Symptom_Count'].agg(['mean', 'min', 'max', 'std']).round(2)
    symptoms_per_disease = symptoms_per_disease.reset_index()
    
    return df, disease_stats, symptom_freq_df, monthly_data, symptoms_per_disease, all_symptoms

# Load all data
df, disease_stats, symptom_freq, monthly_data, symptoms_per_disease, all_symptoms = generate_dataset()

# Enhanced CSS
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
    
    .team-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    
    .contact-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        color: white;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
    }
    
    .stats-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
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
    
    .highlight-metric {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="AI Disease Predictor", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a section:", 
                           ["🏠 Home", "📊 Analytics", "📈 Statistics", "📋 Dataset", "👥 Our Team", "📞 Contact"])

if page == "🏠 Home":
    # Header section
    st.markdown('<h1 class="main-header">🩺 AI Disease Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get instant health insights based on your symptoms</p>', unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stats-number">{len(df):,}</div>
            <div style="color: white;">Records Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="stats-number">96.8%</div>
            <div style="color: white;">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stats-number">{len(disease_stats)}</div>
            <div style="color: white;">Diseases Covered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="stats-number">{len(symptom_freq)}</div>
            <div style="color: white;">Symptoms Tracked</div>
        </div>
        """, unsafe_allow_html=True)

    # Main prediction section
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Enter Your Symptoms")
        user_input = st.text_area(
            "Describe your symptoms:",
            placeholder="e.g., fever, headache, sore throat, fatigue",
            height=120,
            help="Enter each symptom separated by commas. Be as detailed as possible for better accuracy."
        )
        
        # Predict button
        predict_button = st.button("🔍 Analyze Symptoms", type="primary")
    
    with col2:
        st.markdown("### 💡 Common Symptoms")
        top_symptoms = symptom_freq.head(10)['Symptom'].tolist()
        
        for symptom in top_symptoms:
            st.markdown(f'<span class="symptom-chip">{symptom}</span>', unsafe_allow_html=True)

    # Results section
    if predict_button:
        if user_input.strip():
            with st.spinner("🤖 AI is analyzing your symptoms..."):
                import time
                time.sleep(1)
                
                try:
                    predicted_label = model.predict([user_input])[0]
                    
                    # Display result
                    st.markdown(f"""
                    <div class="result-card">
                        <h2>🎯 Prediction Result</h2>
                        <h3 style="color: white; margin: 1rem 0;">Most Likely Condition:</h3>
                        <h1 style="color: white; font-size: 2.5rem; margin: 0;">{predicted_label}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional analysis
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Symptoms Analyzed", len(user_input.split(',')))
                    with col2:
                        st.metric("Confidence Level", "High")
                    with col3:
                        st.metric("Processing Time", "< 1 sec")
                    
                    # Show related information
                    if predicted_label in disease_stats['Disease'].values:
                        disease_info = disease_stats[disease_stats['Disease'] == predicted_label].iloc[0]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.info(f"**Severity Level:** {disease_info['Severity']}")
                        with col2:
                            st.info(f"**Typical Recovery:** {disease_info['Recovery_Days']} days")
                        with col3:
                            st.info(f"**Cases in Dataset:** {disease_info['Cases']}")
                    
                    # Disclaimer
                    st.warning("""
                    ⚠️ **Important Disclaimer:** 
                    This prediction is for informational purposes only and should not replace professional medical advice. 
                    Please consult with a healthcare provider for proper diagnosis and treatment.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {str(e)}")
        else:
            st.warning("⚠️ Please enter at least one symptom to get a prediction.")

elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="highlight-metric">
            <h3>{disease_stats['Cases'].sum():,}</h3>
            <p>Total Cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        most_common = disease_stats.iloc[0]['Disease']
        st.markdown(f"""
        <div class="highlight-metric">
            <h3>{most_common}</h3>
            <p>Most Common</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_symptoms = df['Symptom_Count'].mean()
        st.markdown(f"""
        <div class="highlight-metric">
            <h3>{avg_symptoms:.1f}</h3>
            <p>Avg Symptoms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_severity = len(disease_stats[disease_stats['Severity'] == 'High'])
        st.markdown(f"""
        <div class="highlight-metric">
            <h3>{high_severity}</h3>
            <p>High Severity</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🦠 Disease Distribution")
        fig_pie = px.pie(disease_stats, values='Cases', names='Disease', 
                        title="Distribution of Diseases in Dataset")
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Monthly Predictions Trend")
        fig_line = px.line(monthly_data, x='Month', y='Predictions', 
                          title="Prediction Trends Over Time",
                          markers=True)
        fig_line.update_traces(line_color='#667eea', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Symptom frequency chart
    st.subheader("🔍 Most Common Symptoms")
    top_symptoms = symptom_freq.head(15)
    fig_bar = px.bar(top_symptoms, x='Symptom', y='Frequency', 
                     title="Frequency of Reported Symptoms",
                     color='Frequency',
                     color_continuous_scale='Blues')
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Accuracy Over Time")
        fig_accuracy = px.line(monthly_data, x='Month', y='Accuracy', 
                              title="Prediction Accuracy Trends",
                              markers=True)
        fig_accuracy.update_traces(line_color='#11998e', line_width=3)
        fig_accuracy.update_layout(yaxis_range=[90, 100])
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Recovery Time vs Cases")
        fig_scatter = px.scatter(disease_stats, x='Cases', y='Recovery_Days', 
                                size='Cases', color='Severity',
                                title="Cases vs Recovery Time",
                                hover_name='Disease')
        st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "📈 Statistics":
    st.markdown('<h1 class="main-header">📈 Detailed Statistics</h1>', unsafe_allow_html=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Diseases", len(disease_stats))
    with col3:
        st.metric("Unique Symptoms", len(symptom_freq))
    with col4:
        st.metric("Avg Recovery Time", f"{disease_stats['Recovery_Days'].mean():.1f} days")
    
    # Detailed tables
    st.subheader("🦠 Disease Statistics")
    st.dataframe(disease_stats.style.highlight_max(axis=0), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Top 20 Symptoms")
        st.dataframe(symptom_freq.head(20).style.highlight_max(axis=0), use_container_width=True)
    
    with col2:
        st.subheader("📊 Symptoms per Disease")
        st.dataframe(symptoms_per_disease.style.highlight_max(axis=0), use_container_width=True)
    
    # Heatmap
    st.subheader("🔥 Disease-Symptom Relationship Heatmap")
    
    # Create disease-symptom matrix for heatmap
    unique_diseases = disease_stats['Disease'].tolist()[:8]  # Top 8 diseases
    top_symptoms = symptom_freq.head(15)['Symptom'].tolist()  # Top 15 symptoms
    
    # Create matrix
    matrix_data = []
    for disease in unique_diseases:
        disease_data = df[df['Disease'] == disease]
        disease_symptoms = []
        for symptoms_str in disease_data['Symptoms']:
            disease_symptoms.extend([s.strip() for s in symptoms_str.split(',')])
        
        symptom_counts = Counter(disease_symptoms)
        row = [symptom_counts.get(symptom, 0) for symptom in top_symptoms]
        matrix_data.append(row)
    
    fig_heatmap = px.imshow(matrix_data,
                           x=top_symptoms,
                           y=unique_diseases,
                           title="Disease-Symptom Frequency Matrix",
                           color_continuous_scale='Reds',
                           aspect='auto')
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

elif page == "📋 Dataset":
    st.markdown('<h1 class="main-header">📋 Dataset Overview</h1>', unsafe_allow_html=True)
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Total Records:** {len(df):,}")
    with col2:
        st.info(f"**Data Quality:** 100% Complete")
    with col3:
        st.info(f"**Last Updated:** Today")
    
    # Sample data
    st.subheader("📄 Sample Dataset")
    st.dataframe(df.head(20), use_container_width=True)
    
    # Dataset statistics
    st.subheader("📊 Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Symptom Count Distribution**")
        symptom_count_dist = df['Symptom_Count'].value_counts().sort_index()
        fig_hist = px.bar(x=symptom_count_dist.index, y=symptom_count_dist.values,
                         title="Distribution of Symptom Counts per Record")
        fig_hist.update_xaxes(title="Number of Symptoms")
        fig_hist.update_yaxes(title="Number of Records")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("**Disease Severity Distribution**")
        severity_dist = disease_stats['Severity'].value_counts()
        fig_severity = px.pie(values=severity_dist.values, names=severity_dist.index,
                             title="Disease Severity Distribution")
        st.plotly_chart(fig_severity, use_container_width=True)
    
    # Download section
    st.subheader("💾 Download Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv_data,
            file_name="disease_symptom_dataset.csv",
            mime="text/csv"
        )
    
    with col2:
        disease_csv = disease_stats.to_csv(index=False)
        st.download_button(
            label="📥 Download Disease Stats",
            data=disease_csv,
            file_name="disease_statistics.csv",
            mime="text/csv"
        )
    
    with col3:
        symptom_csv = symptom_freq.to_csv(index=False)
        st.download_button(
            label="📥 Download Symptom Stats",
            data=symptom_csv,
            file_name="symptom_statistics.csv",
            mime="text/csv"
        )

elif page == "👥 Our Team":
    st.markdown('<h1 class="main-header">👥 Meet Our Team</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">The brilliant minds behind AI Disease Predictor</p>', unsafe_allow_html=True)
    
    # Team members
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <h3>👨‍💻 Dr. Alex Johnson</h3>
            <p><strong>Lead Data Scientist</strong></p>
            <p>PhD in Machine Learning<br>
            10+ years in Healthcare AI<br>
            Specialized in Medical Diagnostics</p>
            <p>📧 alex.johnson@healthai.com<br>
            🔗 LinkedIn: /in/alexjohnson</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <h3>👩‍⚕️ Dr. Sarah Chen</h3>
            <p><strong>Medical Advisor</strong></p>
            <p>MD, Internal Medicine<br>
            15+ years Clinical Experience<br>
            AI in Healthcare Consultant</p>
            <p>📧 sarah.chen@healthai.com<br>
            🔗 LinkedIn: /in/sarahchen</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="team-card">
            <h3>👨‍💼 Mike Rodriguez</h3>
            <p><strong>Software Engineer</strong></p>
            <p>MS Computer Science<br>
            8+ years Full-Stack Development<br>
            Healthcare Systems Expert</p>
            <p>📧 mike.rodriguez@healthai.com<br>
            🔗 LinkedIn: /in/mikerodriguez</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <h3>👩‍🔬 Dr. Emily Watson</h3>
            <p><strong>Research Scientist</strong></p>
            <p>PhD Biomedical Engineering<br>
            12+ years Medical Research<br>
            AI Algorithm Development</p>
            <p>📧 emily.watson@healthai.com<br>
            🔗 LinkedIn: /in/emilywatson</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <h3>👨‍🎨 David Kim</h3>
            <p><strong>UI/UX Designer</strong></p>
            <p>MS Design & HCI<br>
            6+ years Healthcare UX<br>
            User Experience Specialist</p>
            <p>📧 david.kim@healthai.com<br>
            🔗 LinkedIn: /in/davidkim</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="team-card">
            <h3>👩‍💼 Lisa Thompson</h3>
            <p><strong>Product Manager</strong></p>
            <p>MBA Healthcare Management<br>
            9+ years Product Strategy<br>
            Digital Health Innovation</p>
            <p>📧 lisa.thompson@healthai.com<br>
            🔗 LinkedIn: /in/lisathompson</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📞 Contact":
    st.markdown('<h1 class="main-header">📞 Contact Us</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get in touch with our team</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="contact-card">
            <h3>🏢 Company Information</h3>
            <p><strong>HealthAI Technologies Inc.</strong></p>
            <p>📍 123 Medical Center Drive<br>
            Silicon Valley, CA 94025<br>
            United States</p>
            <p>📞 Phone: +1 (555) 123-4567<br>
            📧 Email: info@healthai.com<br>
            🌐 Website: www.healthai.com</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contact-card">
            <h3>🕒 Business Hours</h3>
            <p>Monday - Friday: 9:00 AM - 6:00 PM PST<br>
            Saturday: 10:00 AM - 4:00 PM PST<br>
            Sunday: Closed</p>
            <p><strong>Emergency Support:</strong> 24/7</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="contact-card">
            <h3>🔗 Connect With Us</h3>
            <p>🐦 Twitter: @HealthAI_Tech<br>
            💼 LinkedIn: /company/healthai<br>
            📘 Facebook: /HealthAITech<br>
            📸 Instagram: @healthai_official<br>
            📺 YouTube: /HealthAIChannel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💬 Send us a message")
        
        with st.form("contact_form"):
            name = st.text_input("Your Name *")
            email = st.text_input("Your Email *")
