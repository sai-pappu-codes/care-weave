import streamlit as st
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import load_skin_model, predict_skin_disease

st.set_page_config(page_title="Skin Disease | MedAI Nexus", page_icon="🔬", layout="centered")

# ── Paths (update these to your saved model locations) ────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_model_final_fixed.h5")
NAMES_PATH = os.path.join(BASE_DIR, "models", "skin_disease_class_names.pkl")

st.title("🔬 Module 1 — Skin Disease Prediction")
st.caption("Upload a skin image. MobileNetV2 classifies it into one of 20 skin disease categories.")

st.info("📌 Model will be downloaded automatically if not found.", icon="ℹ️")

# ── Load model (cached — only loads once per session) ─────────────────────────
@st.cache_resource(show_spinner="Loading skin model...")
def get_skin_model():
    return load_skin_model(MODEL_PATH, NAMES_PATH)


model, class_names = get_skin_model()

st.write("Model path used:", MODEL_PATH)
st.write("Model loaded:", model is not None)
st.write("Classes loaded:", class_names is not None)

if model is None:
    st.error("❌ Could not load model. Check logs.")
    st.stop()

# ── Upload image ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a skin image (JPEG or PNG)", type=["jpg","jpeg","png"])

if uploaded:
    from PIL import Image
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        with st.spinner("Analysing image…"):
            results = predict_skin_disease(img, model, class_names, top_k=3)

        if results:
            st.markdown("#### Prediction Results")

            top_class, top_conf = results[0]

            # Colour the confidence badge
            if top_conf >= 70:
                badge_color = "🟢"
            elif top_conf >= 45:
                badge_color = "🟡"
            else:
                badge_color = "🔴"

            st.markdown(f"**Top Prediction:** {badge_color} `{top_class}`")
            st.metric("Confidence", f"{top_conf:.1f}%")

            st.markdown("**Top 3 Predictions:**")
            for i, (cls, conf) in enumerate(results, 1):
                bar_len = int(conf / 5)
                st.markdown(
                    f"`{i}.` **{cls[:40]}**  \n"
                    f"{'█' * bar_len}{'░' * (20 - bar_len)} {conf:.1f}%"
                )

            # Save result to session_state for chatbot and coach
            st.session_state['skin_result'] = f"{top_class} ({top_conf:.1f}% confidence)"
            st.success("✓ Result saved — available in AI Chatbot and Lifestyle Coach")

        else:
            st.error("Prediction failed. Please try a different image.")

    st.divider()
    st.warning(
        "⚠️ **Disclaimer:** This is a student AI project. "
        "Results are NOT a medical diagnosis. "
        "Always consult a dermatologist for skin conditions.",
        icon="⚠️"
    )
