import streamlit as st

st.set_page_config(
    page_title="MedAI Nexus",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1A56A0;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .module-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .module-icon { font-size: 2.2rem; }
    .module-title { font-weight: 600; color: #1E293B; margin: 0.4rem 0; }
    .module-desc  { font-size: 0.85rem; color: #64748B; }
    .disclaimer-box {
        background: #FEF9C3;
        border-left: 4px solid #EAB308;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #78350F;
        margin-top: 2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 MedAI Nexus</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Multi-Modal Healthcare Assistant</div>', unsafe_allow_html=True)

st.divider()

# ── Intro ─────────────────────────────────────────────────────────────────────
col_intro, col_status = st.columns([2, 1])

with col_intro:
    st.markdown("""
    **MedAI Nexus** is a student healthcare AI project that combines five intelligent modules:
    upload a skin photo, answer health questions, scan a lab report, chat with an AI doctor,
    and receive a personalised lifestyle plan — all in one place.

    > Use the **sidebar** to navigate between modules.
    > Results from earlier modules automatically personalise later ones.
    """)

with col_status:
    st.markdown("#### Session Status")
    skin_done   = 'skin_result'   in st.session_state
    risk_done   = 'risk_result'   in st.session_state
    report_done = 'report_result' in st.session_state

    def badge(done, label):
        color = "#DCFCE7" if done else "#F1F5F9"
        text_color = "#14532D" if done else "#64748B"
        icon = "✓" if done else "○"
        st.markdown(
            f'<span class="status-badge" style="background:{color};color:{text_color}">'
            f'{icon} {label}</span>', unsafe_allow_html=True
        )
        st.write("")

    badge(skin_done,   "Skin Scan")
    badge(risk_done,   "Health Risk")
    badge(report_done, "Lab Report")

st.divider()

# ── Module Cards ──────────────────────────────────────────────────────────────
st.markdown("### Modules")
cols = st.columns(5)

modules = [
    ("🔬", "Skin Disease",    "Upload a skin photo for disease classification",        "pages/1_Skin_Disease.py"),
    ("💉", "Health Risk",     "Predict diabetes risk from 21 health indicators",       "pages/2_Health_Risk.py"),
    ("📋", "Report Explainer","Scan a lab report and get plain-language explanation",  "pages/3_Report_Explainer.py"),
    ("💬", "AI Chatbot",      "Ask personalised health questions to MedBot",           "pages/4_AI_Chatbot.py"),
    ("🌿", "Lifestyle Coach", "Get a 7-day personalised wellness plan",                "pages/5_Lifestyle_Coach.py"),
]

for col, (icon, title, desc, _) in zip(cols, modules):
    with col:
        st.markdown(f"""
        <div class="module-card">
            <div class="module-icon">{icon}</div>
            <div class="module-title">{title}</div>
            <div class="module-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── How to use ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### How to Use")
steps = [
    "**Step 1** — Go to **Skin Disease** → upload a photo of your skin",
    "**Step 2** — Go to **Health Risk** → fill in your health indicators",
    "**Step 3** — Go to **Report Explainer** → upload your lab report image",
    "**Step 4** — Go to **AI Chatbot** → ask MedBot about your results",
    "**Step 5** — Go to **Lifestyle Coach** → fill the lifestyle form and get your 7-day plan",
]
for s in steps:
    st.markdown(f"- {s}")

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
⚠️ <strong>Medical Disclaimer:</strong>
MedAI Nexus is a student project for educational purposes only.
All outputs are <strong>not medical diagnoses</strong>.
Always consult a qualified healthcare professional before making any health decisions.
</div>
""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#94A3B8;font-size:0.8rem;">'
    'MedAI Nexus · Student Project · Powered by MobileNetV2 · XGBoost · Tesseract · Gemini 1.5 Flash'
    '</div>',
    unsafe_allow_html=True
)
