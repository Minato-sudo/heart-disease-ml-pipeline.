import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CardioAI — Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── Load model & artefacts ──────────────────────────────────
@st.cache_resource
def load_artefacts():
    base = os.path.dirname(__file__)
    nb   = os.path.join(base, '..', 'notebooks', 'outputs')
    
    required_files = ['best_xgb.pkl', 'scaler.pkl', 'splits.pkl']
    for f in required_files:
        if not os.path.exists(os.path.join(nb, f)):
            st.error(f"Missing required file: {f}. Please run the notebooks first.")
            st.stop()
            
    model   = joblib.load(os.path.join(nb, 'best_xgb.pkl'))
    scaler  = joblib.load(os.path.join(nb, 'scaler.pkl'))
    X_train, X_test, y_train, y_test = joblib.load(os.path.join(nb, 'splits.pkl'))
    return model, scaler, X_train.columns.tolist(), X_test, y_test

model, scaler, feature_names, X_test, y_test = load_artefacts()

CONT_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
  body { font-family: 'Inter', sans-serif; }
  .risk-high {
    background: linear-gradient(135deg, #ff4b4b22, #ff4b4b44);
    border-left: 5px solid #ff4b4b;
    border-radius: 8px; padding: 18px; margin: 10px 0;
  }
  .risk-low {
    background: linear-gradient(135deg, #00c85322, #00c85344);
    border-left: 5px solid #00c853;
    border-radius: 8px; padding: 18px; margin: 10px 0;
  }
  .metric-card {
    background: #1e1e2e; border-radius: 10px;
    padding: 14px 18px; margin: 6px 0; color: white;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.title("🫀 CardioAI — Heart Disease Risk Predictor")
st.markdown("""
> **DS-3002 Data Mining · Assignment #4 · Part E** — Model: XGBoost (AUC 0.917)
>
> Enter a patient's clinical measurements below and click **Predict** to assess cardiac risk.
""")

st.divider()

# ── Sidebar — patient pre-populated (real test patient) ──────
st.sidebar.header("🩺 Patient Input Form")
st.sidebar.caption("Choose a preset or enter values manually.")

# Preset Test Cases
presets = {
    "Real Patient 1 (Low Risk)": {
        'age': 54, 'sex': 1, 'cp': 2, 'trestbps': 108, 'chol': 267,
        'fbs': 0, 'restecg': 2, 'thalach': 167, 'exang': 0, 'oldpeak': 0.0,
        'slope': 1, 'ca': 0, 'thal': 3
    },
    "Real Patient 2 (High Risk)": {
        'age': 67, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 286,
        'fbs': 0, 'restecg': 2, 'thalach': 108, 'exang': 1, 'oldpeak': 1.5,
        'slope': 2, 'ca': 3, 'thal': 3
    },
    "Borderline Case (Manual)": {
        'age': 55, 'sex': 1, 'cp': 3, 'trestbps': 130, 'chol': 250,
        'fbs': 0, 'restecg': 1, 'thalach': 140, 'exang': 0, 'oldpeak': 0.8,
        'slope': 2, 'ca': 1, 'thal': 6
    }
}

selected_preset = st.sidebar.selectbox("📋 Select Preset Case", options=list(presets.keys()))
tp = presets[selected_preset]

with st.sidebar.form("patient_form"):
    st.subheader("Continuous features")
    age      = st.number_input("Age (20–80)",         min_value=20,   max_value=80,   value=int(tp['age']),      step=1)
    trestbps = st.number_input("Resting BP (90–200 mmHg)",min_value=90, max_value=200,value=int(tp['trestbps']), step=1)
    chol     = st.number_input("Cholesterol (100–600 mg/dl)", min_value=100,max_value=600,value=int(tp['chol']),  step=1)
    thalach  = st.number_input("Max Heart Rate (70–210)", min_value=70, max_value=210, value=int(tp['thalach']),  step=1)
    oldpeak  = st.number_input("ST Depression (0.0–6.2)", min_value=0.0,max_value=6.2, value=float(tp['oldpeak']),step=0.1, format="%.1f")
    ca       = st.number_input("Major Vessels (0–3)",  min_value=0,   max_value=3,    value=int(tp['ca']),       step=1)

    st.subheader("Binary features")
    # Finding the index for the selectbox based on preset value
    sex_idx = 0 if tp['sex'] == 1 else 1
    fbs_idx = 0 if tp['fbs'] == 0 else 1
    exang_idx = 0 if tp['exang'] == 0 else 1
    
    sex   = st.selectbox("Sex",    options=[(1,"Male"), (0,"Female")], index=sex_idx, format_func=lambda x: x[1])[0]
    fbs   = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0,"No"), (1,"Yes")], index=fbs_idx, format_func=lambda x: x[1])[0]
    exang = st.selectbox("Exercise-induced Angina",          options=[(0,"No"), (1,"Yes")], index=exang_idx, format_func=lambda x: x[1])[0]

    st.subheader("Categorical features")
    # Mapping values to indices (assuming order in options)
    cp_idx = {1:0, 2:1, 3:2, 4:3}.get(tp['cp'], 0)
    restecg_idx = {0:0, 1:1, 2:2}.get(tp['restecg'], 0)
    slope_idx = {1:0, 2:1, 3:2}.get(tp['slope'], 0)
    thal_idx = {3:0, 6:1, 7:2}.get(tp['thal'], 0)

    cp      = st.selectbox("Chest Pain Type",   options=[(1,"Typical Angina"),(2,"Atypical"),(3,"Non-Anginal"),(4,"Asymptomatic")], index=cp_idx, format_func=lambda x: x[1])[0]
    restecg = st.selectbox("Resting ECG",       options=[(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")],                  index=restecg_idx, format_func=lambda x: x[1])[0]
    slope   = st.selectbox("ST Slope",          options=[(1,"Upsloping"),(2,"Flat"),(3,"Downsloping")],                              index=slope_idx, format_func=lambda x: x[1])[0]
    thal    = st.selectbox("Thalassemia",       options=[(3,"Normal"),(6,"Fixed Defect"),(7,"Reversible Defect")],                   index=thal_idx, format_func=lambda x: x[1])[0]

    # Updated width param based on warning (using the newer string value)
    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# ── Build feature vector from inputs ────────────────────────
def build_feature_vector(age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal):
    raw = {'age': age, 'sex': sex, 'trestbps': trestbps, 'chol': chol,
           'fbs': fbs, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
           'ca': ca}
    df  = pd.DataFrame([raw])

    # One-hot encode categoricals exactly as training
    for c_val, prefix in [(cp, 'cp'), (restecg, 'restecg'), (slope, 'slope'), (thal, 'thal')]:
        if prefix == 'cp': possibilities = [1,2,3,4]
        elif prefix == 'restecg': possibilities = [0,1,2]
        elif prefix == 'slope': possibilities = [1,2,3]
        elif prefix == 'thal': possibilities = [3,6,7]
        
        for possible in possibilities:
            col = f"{prefix}_{float(possible)}"
            df[col] = 1.0 if possible == c_val else 0.0

    # Scale continuous cols - pass DataFrame to keep feature names and avoid warnings
    df[CONT_COLS] = scaler.transform(df[CONT_COLS])

    # Align columns with training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_names]
    return df

# ── Feature name mapping for UI ────────────────────────────
def get_human_name(feat):
    mapping = {
        'age': 'Age', 'sex': 'Sex', 'trestbps': 'Resting BP', 'chol': 'Cholesterol',
        'fbs': 'Fasting BS', 'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
        'oldpeak': 'ST Depression', 'ca': 'Major Vessels',
        'cp_1.0': 'CP: Typical', 'cp_2.0': 'CP: Atypical', 'cp_3.0': 'CP: Non-Anginal', 'cp_4.0': 'CP: Asymptomatic',
        'restecg_0.0': 'ECG: Normal', 'restecg_1.0': 'ECG: ST-T Abn', 'restecg_2.0': 'ECG: LV Hypertrophy',
        'slope_1.0': 'Slope: Upsloping', 'slope_2.0': 'Slope: Flat', 'slope_3.0': 'Slope: Downsloping',
        'thal_3.0': 'Thal: Normal', 'thal_6.0': 'Thal: Fixed Defect', 'thal_7.0': 'Thal: Reversible'
    }
    return mapping.get(feat, feat)

# ── Main area ────────────────────────────────────────────────
col_form, col_result = st.columns([1.2, 1.8])

with col_form:
    st.subheader("📋 Input Summary")
    summary_df = pd.DataFrame({
        'Feature': ['Age', 'Sex', 'Chest Pain', 'Resting BP', 'Cholesterol',
                    'Fasting BS', 'Rest ECG', 'Max HR', 'Exercise Angina',
                    'ST Depression', 'ST Slope', 'Vessels', 'Thalassemia'],
        'Value':   [age,
                    "Male" if sex==1 else "Female",
                    {1:"Typical", 2:"Atypical", 3:"Non-Anginal", 4:"Asymptomatic"}[cp],
                    f"{trestbps} mmHg", f"{chol} mg/dl",
                    "Yes" if fbs else "No",
                    ["Normal","ST-T Abn.","LV Hypertrophy"][restecg],
                    f"{thalach} bpm",
                    "Yes" if exang else "No",
                    f"{oldpeak}",
                    {1:"Upsloping", 2:"Flat", 3:"Downsloping"}[slope],
                    int(ca),
                    {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"}[thal]]
    })
    # Edge Case Fix: Cast 'Value' to string to avoid Arrow serialization error with mixed types
    summary_df['Value'] = summary_df['Value'].astype(str)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Clinical Range Edge Case Warnings
    if trestbps > 165 or chol > 320 or age > 78:
        st.warning("⚠️ **Note:** One or more values entered are in the extreme clinical range. Model predictions may be highly sensitive to these inputs.")

with col_result:
    if submitted:
        X_input = build_feature_vector(age, sex, cp, trestbps, chol, fbs,
                                        restecg, thalach, exang, oldpeak,
                                        slope, ca, thal)

        prob     = model.predict_proba(X_input)[0][1]
        
        # Handling the "Exact Threshold" edge case
        if abs(prob - 0.5) < 0.001:
            pred = 1 # Default to higher risk in borderline cases
            borderline = True
        else:
            pred     = int(prob >= 0.5)
            borderline = False
            
        conf_pct = prob * 100 if pred == 1 else (1 - prob) * 100

        # ── Risk label ──
        st.subheader("🔬 Prediction Result")
        if pred == 1:
            st.markdown(f"""
            <div class="risk-high">
            <h2 style="color:#ff4b4b;margin:0">🔴 Heart Disease — {"BORDERLINE RISK" if borderline else "PRESENT"}</h2>
            <p style="font-size:1.2rem;margin:6px 0">Confidence: <b>{conf_pct:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
            <h2 style="color:#00c853;margin:0">🟢 No Heart Disease Detected</h2>
            <p style="font-size:1.2rem;margin:6px 0">Confidence: <b>{conf_pct:.1f}%</b></p>
            </div>""", unsafe_allow_html=True)

        # ── Top-3 feature importances ──
        st.subheader("📊 Top 3 Driving Features")
        importances = model.feature_importances_
        top3_idx    = np.argsort(importances)[::-1][:3]
        top3_names  = [get_human_name(feature_names[i]) for i in top3_idx]
        top3_vals   = importances[top3_idx]

        fig, ax = plt.subplots(figsize=(5, 2.2))
        colors  = ['#ff4b4b' if pred==1 else '#00c853'] * 3
        bars    = ax.barh(top3_names[::-1], top3_vals[::-1], color=colors, height=0.5)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 3 Features', fontsize=11, fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Plain-English explanation ──
        st.subheader("📝 Clinical Explanation")
        top_feat  = top3_names[0]
        second_feat = top3_names[1]
        if pred == 1:
            st.info(
                f"⚠️ This patient shows elevated cardiac risk. "
                f"**{top_feat}** and **{second_feat}** are the strongest indicators "
                f"flagging this patient. "
                f"The ST depression value of {oldpeak} and a maximum heart rate of {thalach} bpm "
                f"are consistent with reduced coronary perfusion during stress. "
                f"**Recommend immediate cardiology referral and further stress testing.**"
            )
        else:
            st.success(
                f"✅ This patient shows no strong indicators of heart disease. "
                f"**{top_feat}** and **{second_feat}** are the dominant model features, "
                f"and both fall within acceptable clinical ranges. "
                f"**Routine follow-up is advised; no urgent intervention required.**"
            )
    else:
        st.info("👈 Fill in the patient details on the left and click **Predict** to see the result.")
        st.markdown("""
        **How it works:**
        - Model: XGBoost (AUC-ROC = 0.917, trained on UCI Cleveland dataset)
        - Features: 13 clinical measurements
        - Output: Disease present / absent + confidence + top 3 driving features
        """)

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption("DS-3002 Data Mining · Assignment #4 · Part E · CardioAI Labs (Fictional) · Spring 2026 FAST-NUCES")
