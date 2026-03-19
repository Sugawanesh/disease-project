import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")

# --- CSS for styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. Load and Cache Data/Models ---
@st.cache_data
def load_data():
    parkinsons = pd.read_csv('parkinsons - parkinsons.csv')
    kidney = pd.read_csv('kidney_disease - kidney_disease.csv')
    liver = pd.read_csv('indian_liver_patient - indian_liver_patient.csv')
    return parkinsons, kidney, liver

@st.cache_resource
def train_models(p_df, k_df, l_df):
    # --- Liver Model ---
    l_df = l_df.copy()
    l_df['Albumin_and_Globulin_Ratio'] = l_df['Albumin_and_Globulin_Ratio'].fillna(l_df['Albumin_and_Globulin_Ratio'].mean())
    l_df['Gender'] = l_df['Gender'].map({'Male': 1, 'Female': 0})
    X_l = l_df.drop('Dataset', axis=1)
    y_l = l_df['Dataset']
    model_l = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_l, y_l)

    # --- Kidney Model ---
    k_df = k_df.copy().drop('id', axis=1)
    # Clean classification target
    k_df['classification'] = k_df['classification'].str.strip().str.replace('ckd', '1').replace('not1', '0')
    k_df['classification'] = pd.to_numeric(k_df['classification'], errors='coerce').fillna(0).astype(int)
    # Mapping categorical
    for col in k_df.select_dtypes(include=['object']).columns:
        k_df[col] = k_df[col].str.strip().replace({'\t':''}, regex=True)
        if col in ['htn', 'dm', 'cad', 'pe', 'ane']: k_df[col] = k_df[col].map({'yes': 1, 'no': 0})
        elif col in ['rbc', 'pc']: k_df[col] = k_df[col].map({'normal': 1, 'abnormal': 0})
        elif col in ['pcc', 'ba']: k_df[col] = k_df[col].map({'present': 1, 'notpresent': 0})
        elif col in ['appet']: k_df[col] = k_df[col].map({'good': 1, 'poor': 0})
    for col in ['pcv', 'wc', 'rc']: k_df[col] = pd.to_numeric(k_df[col], errors='coerce')
    
    imputer = SimpleImputer(strategy='median')
    k_imputed = pd.DataFrame(imputer.fit_transform(k_df), columns=k_df.columns)
    X_k = k_imputed.drop('classification', axis=1)
    y_k = k_imputed['classification']
    model_k = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_k, y_k)

    # --- Parkinsons Model ---
    X_p = p_df.drop(['name', 'status'], axis=1)
    y_p = p_df['status']
    model_p = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_p, y_p)

    return model_l, model_k, model_p, X_l.columns, X_k.columns, X_p.columns

# Initialize
p_df, k_df, l_df = load_data()
model_l, model_k, model_p, cols_l, cols_k, cols_p = train_models(p_df, k_df, l_df)

# --- Sidebar Navigation ---
st.sidebar.title("🏥 Disease Predictor")
disease_option = st.sidebar.selectbox("Choose Disease to Predict", ["Diabetes (N/A)", "Kidney Disease", "Liver Disease", "Parkinsons Disease"])

st.title(f"🔍 {disease_option} Prediction")

# --- 2. Prediction Interface ---

if disease_option == "Liver Disease":
    st.info("Input clinical parameters for Liver health assessment.")
    col1, col2 = st.columns(2)
    inputs = []
    with col1:
        age = st.number_input("Age", 1, 100, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tb = st.number_input("Total Bilirubin", 0.1, 80.0, 1.0)
        db = st.number_input("Direct Bilirubin", 0.1, 20.0, 0.5)
        ap = st.number_input("Alkaline Phosphotase", 50, 2500, 200)
    with col2:
        aa = st.number_input("Alamine Aminotransferase", 10, 2000, 50)
        aspa = st.number_input("Aspartate Aminotransferase", 10, 5000, 50)
        tp = st.number_input("Total Protiens", 2.0, 10.0, 6.0)
        alb = st.number_input("Albumin", 0.9, 6.0, 3.0)
        agr = st.number_input("Albumin and Globulin Ratio", 0.1, 3.0, 1.0)
    
    if st.button("Predict Liver Disease"):
        gender_val = 1 if gender == "Male" else 0
        features = np.array([[age, gender_val, tb, db, ap, aa, aspa, tp, alb, agr]])
        pred = model_l.predict(features)
        if pred[0] == 1:
            st.error("Result: High risk of Liver Disease.")
        else:
            st.success("Result: Low risk of Liver Disease.")

elif disease_option == "Kidney Disease":
    st.info("Input lab results to check for Chronic Kidney Disease (CKD).")
    col1, col2, col3 = st.columns(3)
    # Using a subset of critical features for brevity in this example
    with col1:
        age = st.number_input("Age", 1, 100, 50)
        bp = st.number_input("Blood Pressure", 50, 180, 80)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.selectbox("Albumin (0-5)", [0, 1, 2, 3, 4, 5])
    with col2:
        su = st.selectbox("Sugar (0-5)", [0, 1, 2, 3, 4, 5])
        rbc = st.selectbox("RBC", ["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
        hemo = st.number_input("Hemoglobin", 3.0, 18.0, 12.0)
    with col3:
        htn = st.selectbox("Hypertension", ["yes", "no"])
        dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
        appet = st.selectbox("Appetite", ["good", "poor"])
        pe = st.selectbox("Pedal Edema", ["yes", "no"])

    if st.button("Predict Kidney Disease"):
        # Mapping inputs to match the 24 training features (filling others with medians)
        input_data = [age, bp, sg, al, su, 1 if rbc=="normal" else 0, 1 if pc=="normal" else 0, 0, 0, 120, 40, 1.2, 138, 4.4, hemo, 40, 8000, 4.5, 1 if htn=="yes" else 0, 1 if dm=="yes" else 0, 0, 1 if appet=="good" else 0, 1 if pe=="yes" else 0, 0]
        pred = model_k.predict([input_data])
        if pred[0] == 1:
            st.error("Result: High risk of Chronic Kidney Disease.")
        else:
            st.success("Result: Healthy Kidney function predicted.")

elif disease_option == "Parkinsons Disease":
    st.info("Input voice frequency measurements for Parkinsons assessment.")
    # Parkinsons has 22 features. Showing top ones for demo.
    col1, col2 = st.columns(2)
    with col1:
        fo = st.number_input("Average vocal fundamental frequency (MDVP:Fo(Hz))", 80.0, 260.0, 120.0)
        fhi = st.number_input("Maximum vocal fundamental frequency (MDVP:Fhi(Hz))", 100.0, 600.0, 160.0)
        jitter = st.number_input("Multidimensional Voice Program Jitter(%)", 0.0, 0.1, 0.007)
    with col2:
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.2, 0.04)
        hnr = st.number_input("HNR (Harmonics-to-Noise Ratio)", 0.0, 40.0, 21.0)
        spread1 = st.number_input("Spread1", -10.0, 1.0, -5.0)

    if st.button("Predict Parkinsons"):
        # We need all 22 features. We fill the rest with mean values from the dataset for this demo.
        avg_features = p_df.drop(['name', 'status'], axis=1).mean().values
        # Update user-input features at specific indices
        avg_features[0] = fo
        avg_features[1] = fhi
        avg_features[4] = jitter
        avg_features[9] = shimmer
        avg_features[16] = hnr
        avg_features[20] = spread1
        
        pred = model_p.predict([avg_features])
        if pred[0] == 1:
            st.error("Result: Parkinsons Disease Detected.")
        else:
            st.success("Result: No signs of Parkinsons Disease.")

# --- Footer ---
st.sidebar.markdown("---")