import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import load_risk_model, predict_diabetes_risk, make_shap_waterfall_fig

st.set_page_config(page_title="Health Risk | MedAI Nexus", page_icon="💉", layout="wide")

MODEL_PATH    = "models/health_risk_xgb.pkl"
FEATURES_PATH = "models/health_features.pkl"

st.title("💉 Module 2 — Health Risk Prediction")
st.caption("Fill in your health indicators. XGBoost predicts your diabetes risk with SHAP explanation.")

@st.cache_resource(show_spinner="Loading health risk model…")
def get_risk_model():
    return load_risk_model(MODEL_PATH, FEATURES_PATH)

model, features = get_risk_model()

if model is None:
    st.error("❌ Could not load model. Check that health_risk_xgb.pkl and health_features.pkl are in models/.")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("health_form"):
    st.markdown("### Your Health Information")
    st.markdown("All fields are from the CDC BRFSS health survey format.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**General Health**")
        gen_hlth = st.selectbox("General Health (1=Excellent, 5=Poor)", [1,2,3,4,5], index=2)
        bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.5)
        age      = st.selectbox("Age Group", list(range(1,14)),
                                format_func=lambda x: {
                                    1:"18-24",2:"25-29",3:"30-34",4:"35-39",5:"40-44",
                                    6:"45-49",7:"50-54",8:"55-59",9:"60-64",10:"65-69",
                                    11:"70-74",12:"75-79",13:"80+"
                                }[x], index=4)
        sex       = st.selectbox("Sex", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
        education = st.selectbox("Education Level (1-6)", [1,2,3,4,5,6], index=3)
        income    = st.selectbox("Income Level (1-8)", [1,2,3,4,5,6,7,8], index=4)

    with col2:
        st.markdown("**Medical Conditions**")
        high_bp   = st.radio("High Blood Pressure?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        high_chol = st.radio("High Cholesterol?",    [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        chol_check= st.radio("Cholesterol Check (last 5 yrs)?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        stroke    = st.radio("Ever had a Stroke?",   [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        heart_dis = st.radio("Heart Disease / Attack?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        diff_walk = st.radio("Difficulty Walking?",  [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

        st.markdown("**Physical Health (last 30 days)**")
        phys_hlth = st.slider("Days of poor physical health (0-30)", 0, 30, 5)
        ment_hlth = st.slider("Days of poor mental health (0-30)", 0, 30, 3)

    with col3:
        st.markdown("**Lifestyle**")
        phys_act = st.radio("Physical Activity (last 30 days)?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        fruits   = st.radio("Eat Fruit daily?",   [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        veggies  = st.radio("Eat Vegetables daily?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        smoker   = st.radio("Smoker (100+ cigarettes lifetime)?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        alcohol  = st.radio("Heavy Alcohol Use?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

        st.markdown("**Healthcare Access**")
        any_hc    = st.radio("Any Health Coverage?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        no_doc    = st.radio("Couldn't see doctor due to cost?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

    submitted = st.form_submit_button("🔍 Predict My Risk", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    patient_data = {
        'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': chol_check,
        'BMI': bmi, 'Smoker': smoker, 'Stroke': stroke,
        'HeartDiseaseorAttack': heart_dis, 'PhysActivity': phys_act,
        'Fruits': fruits, 'Veggies': veggies, 'HvyAlcoholConsump': alcohol,
        'AnyHealthcare': any_hc, 'NoDocbcCost': no_doc,
        'GenHlth': gen_hlth, 'MentHlth': ment_hlth, 'PhysHlth': phys_hlth,
        'DiffWalk': diff_walk, 'Sex': sex, 'Age': age,
        'Education': education, 'Income': income,
    }

    with st.spinner("Calculating risk…"):
        result = predict_diabetes_risk(patient_data, model, features)

    st.divider()
    st.markdown("## Your Risk Assessment")

    # Risk level display
    risk    = result['risk_level']
    prob    = result['probability']
    color_map = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}

    col_res, col_gauge = st.columns([1, 1])
    with col_res:
        st.markdown(f"### {color_map[risk]} Risk Level: **{risk}**")
        st.metric("Probability of Diabetes", f"{prob:.1f}%")
        if risk == "High":
            st.error("Your risk is elevated. Please consult a doctor soon.")
        elif risk == "Moderate":
            st.warning("Moderate risk detected. Lifestyle changes are recommended.")
        else:
            st.success("Low risk. Keep up your healthy habits!")

    with col_gauge:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 1.5))
        color_vals = {"Low":"#22c55e","Moderate":"#f59e0b","High":"#ef4444"}
        ax.barh([""], [prob], color=color_vals[risk], height=0.4)
        ax.barh([""], [100 - prob], left=[prob], color="#E2E8F0", height=0.4)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 30, 60, 100])
        ax.set_xticklabels(["0%","Low","Moderate","100%"], fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"Risk: {prob:.1f}%", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # SHAP top factors
    if result['top_factors']:
        st.markdown("#### Top Contributing Factors (SHAP)")
        for feat, shap_val in result['top_factors']:
            direction = "↑ Increases risk" if shap_val > 0 else "↓ Decreases risk"
            color_dir = "🔴" if shap_val > 0 else "🟢"
            st.markdown(f"- {color_dir} **{feat}**: {direction} (SHAP = {shap_val:+.4f})")

    # SHAP waterfall
    with st.expander("📊 Show detailed SHAP waterfall chart"):
        fig = make_shap_waterfall_fig(patient_data, model, features)
        if fig:
            st.pyplot(fig)
            plt.close()
        else:
            st.info("SHAP chart requires the shap library. Install with: pip install shap")

    # Save to session state
    st.session_state['risk_result'] = f"{risk} risk ({prob:.1f}% probability)"
    st.success("✓ Result saved — available in AI Chatbot and Lifestyle Coach")

    st.divider()
    st.warning("⚠️ This is a student AI project. Not a medical diagnosis. Consult a doctor for diabetes screening.", icon="⚠️")
