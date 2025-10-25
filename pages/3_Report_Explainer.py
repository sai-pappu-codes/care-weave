import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import extract_ocr_text, explain_report_gemini

st.set_page_config(page_title="Report Explainer | MedAI Nexus", page_icon="📋", layout="wide")

st.title("📋 Module 3 — Medical Report Explainer")
st.caption("Upload a lab report image. Tesseract reads it, Gemini 1.5 Flash explains it in plain language.")

# ── API Key ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("Enter your Gemini API key", type="password",
                             help="Free at: https://aistudio.google.com/app/apikey")
    st.caption("Key is used only in your session and never stored.")

if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to enable explanations.", icon="🔑")

# ── Patient context ───────────────────────────────────────────────────────────
with st.expander("Optional: Add patient context for better explanations"):
    col_a, col_b = st.columns(2)
    with col_a:
        patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=35, step=1)
    with col_b:
        patient_gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])
    if patient_gender == "Not specified":
        patient_gender = None

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload lab report image (JPEG, PNG) or PDF page screenshot",
    type=["jpg","jpeg","png"]
)

if uploaded:
    from PIL import Image
    img = Image.open(uploaded).convert("RGB")

    col_img, col_text = st.columns([1, 1])

    with col_img:
        st.image(img, caption="Uploaded Report", use_container_width=True)

    with col_text:
        # Step 1: OCR
        with st.spinner("Step 1: Extracting text via OCR…"):
            ocr_text = extract_ocr_text(img)

        if not ocr_text or len(ocr_text.strip()) < 20:
            st.error("Could not extract enough text. Please upload a clearer image.")
            st.stop()

        word_count = len(ocr_text.split())
        st.success(f"✓ OCR extracted {word_count} words")

        with st.expander("Show raw OCR text"):
            st.text(ocr_text[:2000] + ("…" if len(ocr_text) > 2000 else ""))

        if not api_key:
            st.warning("Add Gemini API key in the sidebar to generate an explanation.")
            st.stop()

        # Step 2: Gemini explanation
        with st.spinner("Step 2: Gemini 1.5 Flash is analysing your report…"):
            result = explain_report_gemini(
                ocr_text, patient_age, patient_gender, api_key
            )

    # ── Display results ───────────────────────────────────────────────────────
    if result.get('parse_error'):
        st.error("Gemini could not parse the report. Raw response below:")
        st.code(result.get('raw_response','')[:1000])
        st.stop()

    st.divider()
    st.markdown("## Report Explanation")

    # Patient summary
    if result.get('patient_summary'):
        st.info(f"📋 **Summary:** {result['patient_summary']}")

    # Flags / alerts
    flags = result.get('flags', [])
    if flags:
        st.error("⚠️ Values needing attention:")
        for flag in flags:
            st.markdown(f"- 🚨 {flag}")

    # Test results table
    tests = result.get('test_results', [])
    if tests:
        st.markdown("### Test Results")
        for test in tests:
            status = test.get('status', 'Unknown')
            color_map = {
                'Normal'   : '🟢',
                'Low'      : '🟡',
                'High'     : '🟡',
                'Critical' : '🔴',
            }
            icon = color_map.get(status, '⚪')

            with st.expander(f"{icon} **{test.get('test_name','—')}** — {test.get('value','')} {test.get('unit','')} ({status})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Reference Range:** {test.get('reference_range','N/A')}")
                    st.markdown(f"**Status:** {status}")
                with c2:
                    st.markdown(f"**What it measures:** {test.get('explanation','N/A')}")
                st.markdown(f"**In simple terms:** {test.get('patient_friendly','N/A')}")

    # Next steps
    next_steps = result.get('next_steps', [])
    if next_steps:
        st.markdown("### Recommended Next Steps")
        for step in next_steps:
            st.markdown(f"- ✅ {step}")

    # Save summary to session state
    summary_text = result.get('patient_summary','')
    if flags:
        summary_text += f" Flags: {'; '.join(flags[:2])}"
    st.session_state['report_result'] = summary_text[:300]
    st.success("✓ Report summary saved — available in AI Chatbot and Lifestyle Coach")

    # Disclaimer
    st.divider()
    st.warning(result.get('disclaimer',
        "⚠️ This analysis is for educational purposes only. Not a medical diagnosis. "
        "Always consult a qualified healthcare professional."), icon="⚠️")
