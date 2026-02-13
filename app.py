import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
from pathlib import Path

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="Slope Stability AI",
    # page_icon="⛰️",
    layout="wide"
)

# ---------------------------
# Set background image
# ---------------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        /* Background image */
        .stApp {{
            background-image: url(data:image/webp;base64,{b64_encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Make input fields transparent with lighter border and smaller size */
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {{
            background: rgba(255, 255, 255, 0.2) !important;  /* very light white transparent */
            background-color: rgba(255, 255, 255, 0.2) !important;
            border: 1px solid rgba(0, 0, 0, 0.15) !important;  /* lighter border */
            color: #ffffff !important;
            height: 2.2rem !important;          /* smaller height */
            font-size: 0.9rem !important;        /* smaller font */
            padding: 0.25rem 0.5rem !important;  /* reduced padding */
            border-radius: 0.3rem !important;    /* slightly rounded corners */
        }}
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background: rgba(255, 255, 255, 0.2) !important;
            background-color: rgba(255, 255, 255, 0.2) !important;
            border: 1px solid rgba(0, 0, 0, 0.15) !important;
            color: #ffffff !important;
            min-height: 2.2rem !important;       /* smaller height */
            font-size: 0.9rem !important;
            padding: 0.25rem 0.5rem !important;
            border-radius: 0.3rem !important;
        }}
        /* Style labels */
        .stNumberInput label, .stSelectbox label {{
            color: #ffffff !important;
            font-weight: 500;
            font-size: 0.9rem !important;        /* smaller label */
            text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
        }}
        /* Remove any background from the main content area */
        .main .block-container {{
            background: transparent !important;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background("background.webp")

# ---------------------------
# Cache loading of artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    encoder = joblib.load("artifacts/data_preprocessing/encoder.joblib")
    scaler = joblib.load("artifacts/data_preprocessing/scaler.joblib")
    model = joblib.load("artifacts/model_trainer/model.joblib")
    return encoder, scaler, model

encoder, scaler, model = load_artifacts()

# ---------------------------
# Feature engineering function
# ---------------------------
def engineer_features(df):
    df = df.copy()
    beta_rad = np.radians(df['Slope Angle (°)'])
    phi_rad = np.radians(df['Internal Friction Angle (°)'])

    df['Stability_Number'] = df['Cohesion (kPa)'] / (df['Unit Weight (kN/m³)'] * df['Slope Height (m)'])
    df['tan_phi'] = np.tan(phi_rad)
    df['Effective_Friction'] = (1 - df['Pore Water Pressure Ratio']) * df['tan_phi']
    df['Height_x_sinbeta'] = df['Slope Height (m)'] * np.sin(beta_rad)
    df['c_over_gamma'] = df['Cohesion (kPa)'] / df['Unit Weight (kN/m³)']
    df['phi_x_ru'] = df['Internal Friction Angle (°)'] * df['Pore Water Pressure Ratio']
    return df

# ---------------------------
# Preprocessing pipeline for a single row
# ---------------------------
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df_fe = engineer_features(df)

    # One‑hot encode Reinforcement Numeric
    encoded = encoder.transform(df_fe[['Reinforcement Numeric']])
    feature_names = encoder.get_feature_names_out(['Reinforcement Numeric'])
    encoded_df = pd.DataFrame(encoded, columns=feature_names)
    df_fe = pd.concat([df_fe.drop('Reinforcement Numeric', axis=1), encoded_df], axis=1)

    # Ensure column order matches training
    if hasattr(scaler, 'feature_names_in_'):
        expected_cols = list(scaler.feature_names_in_)
        for col in expected_cols:
            if col not in df_fe.columns:
                df_fe[col] = 0
        df_fe = df_fe[expected_cols]

    scaled = scaler.transform(df_fe)
    return scaled

# ---------------------------
# Sidebar – Information
# ---------------------------
st.sidebar.image("124661-fugro.avif", width=200)
st.sidebar.title("Slope Stability AI")
st.sidebar.info(
    """
    This tool predicts whether a slope is **STABLE** or **UNSTABLE** based on geotechnical parameters.
    The model was trained on 10,000 synthetic slope cases and achieves **>98% accuracy**.
    """
)
# st.sidebar.markdown("---")
# st.sidebar.subheader("📊 Model Performance")
# st.sidebar.metric("Accuracy", "98.5%", "XGBoost")
# st.sidebar.metric("Recall (Unstable)", "99.1%", "high safety")
# st.sidebar.markdown("---")
# st.sidebar.caption("Created for Fugro AI Student Assessment")

# ---------------------------
# Main panel
# ---------------------------
st.title("Fugro Slope Stability Assessment")
st.markdown("Enter the slope parameters below to get an instant safety evaluation.")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    unit_weight = st.number_input("Unit Weight (kN/m³)", min_value=15.0, max_value=25.0, value=20.0, step=0.1)
    cohesion = st.number_input("Cohesion (kPa)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    friction_angle = st.number_input("Internal Friction Angle (°)", min_value=20.0, max_value=45.0, value=32.0, step=0.1)
    slope_angle = st.number_input("Slope Angle (°)", min_value=10.0, max_value=60.0, value=35.0, step=0.1)

with col2:
    slope_height = st.number_input("Slope Height (m)", min_value=5.0, max_value=50.0, value=25.0, step=0.1)
    pore_pressure = st.number_input("Pore Water Pressure Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    reinforcement = st.selectbox("Reinforcement Type", options=[0, 1, 2, 3],
                                 format_func=lambda x: ["Retaining Wall", "Soil Nailing", "Geosynthetics", "Drainage"][x])

# Assemble input dictionary
input_data = {
    'Unit Weight (kN/m³)': unit_weight,
    'Cohesion (kPa)': cohesion,
    'Internal Friction Angle (°)': friction_angle,
    'Slope Angle (°)': slope_angle,
    'Slope Height (m)': slope_height,
    'Pore Water Pressure Ratio': pore_pressure,
    'Reinforcement Numeric': reinforcement
}

# ---------------------------
# Prediction button
# ---------------------------
if st.button("Assess Safety", type="primary"):
    with st.spinner("Running geotechnical analysis..."):
        X_scaled = preprocess_input(input_data)

        prob = model.predict_proba(X_scaled)[0, 1]
        pred = model.predict(X_scaled)[0]
        status = "STABLE" if pred == 1 else "UNSTABLE"
        color = "green" if pred == 1 else "red"

        # Display result
        st.markdown(f"## <span style='color:{color};'>⛰️ {status}</span>", unsafe_allow_html=True)
        st.progress(float(prob))
        st.metric("Confidence", f"{prob:.1%}")

        # Two‑column layout: SHAP + Gauge
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Shap Values")
            # st.markdown("Feature contributions (SHAP)")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
                base_val = explainer.expected_value[1]
            else:
                shap_vals = shap_values[0]
                base_val = explainer.expected_value

            explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_val,
                data=X_scaled[0],
                feature_names=scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else [f"f{i}" for i in range(X_scaled.shape[1])]
            )

            fig, ax = plt.subplots(figsize=(6, 3))
            shap.waterfall_plot(explanation, max_display=8, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close()

        with col_right:
            st.subheader("Safety Gauge")
            # st.markdown("Probability of stability")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 30}},
                title={'text': "Stability Probability", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ffcccc'},
                        {'range': [50, 70], 'color': '#ffffcc'},
                        {'range': [70, 100], 'color': '#ccffcc'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))

            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="white",
                font={'color': "darkgray", 'family': "Arial"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption("🔴 Unstable (<50%) | 🟡 Marginal (50‑70%) | 🟢 Stable (>70%)")

        with st.expander("📋 Input Summary"):
            st.json(input_data)

# ---------------------------
# Batch upload (bonus)
# ---------------------------
st.markdown("---")
st.subheader("📁 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with multiple slopes (same columns as above)", type="csv")
if uploaded_file is not None:
    df_batch = pd.read_csv(uploaded_file)
    st.write("Preview:", df_batch.head())

    if st.button("Run Batch Assessment"):
        with st.spinner("Processing..."):
            df_fe = engineer_features(df_batch)
            encoded = encoder.transform(df_fe[['Reinforcement Numeric']])
            feature_names = encoder.get_feature_names_out(['Reinforcement Numeric'])
            encoded_df = pd.DataFrame(encoded, columns=feature_names)
            df_fe = pd.concat([df_fe.drop('Reinforcement Numeric', axis=1), encoded_df], axis=1)

            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = list(scaler.feature_names_in_)
                for col in expected_cols:
                    if col not in df_fe.columns:
                        df_fe[col] = 0
                df_fe = df_fe[expected_cols]

            X_scaled_batch = scaler.transform(df_fe)
            probs = model.predict_proba(X_scaled_batch)[:, 1]
            preds = model.predict(X_scaled_batch)

            results = df_batch.copy()
            results['Safety_Status'] = preds
            results['Safety_Assessment'] = results['Safety_Status'].map({1: 'STABLE', 0: 'UNSTABLE'})
            results['Confidence'] = probs

            st.success("Batch assessment complete!")
            st.dataframe(results)
            csv = results.to_csv(index=False)
            st.download_button("Download Results", data=csv, file_name="slope_predictions.csv", mime="text/csv")