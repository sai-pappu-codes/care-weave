import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import create_chat_session, send_chat_message, is_emergency

st.set_page_config(page_title="AI Chatbot | MedAI Nexus", page_icon="💬", layout="centered")

st.title("💬 Module 4 — MedBot AI Health Chatbot")
st.caption("Ask health questions. MedBot answers using your results from Modules 1, 2, and 3.")

# ── Sidebar: API key + context preview ────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("Enter your Gemini API key", type="password",
                             key="chatbot_api_key",
                             help="Free at: https://aistudio.google.com/app/apikey")
    st.divider()
    st.markdown("### Your Health Context")
    skin_r   = st.session_state.get('skin_result',   'Not assessed yet')
    risk_r   = st.session_state.get('risk_result',   'Not assessed yet')
    report_r = st.session_state.get('report_result', 'Not assessed yet')
    st.markdown(f"🔬 **Skin:** {skin_r[:50]}")
    st.markdown(f"💉 **Risk:** {risk_r[:50]}")
    st.markdown(f"📋 **Report:** {report_r[:60]}")
    st.caption("Complete Modules 1–3 for personalised answers.")
    st.divider()
    if st.button("🔄 Reset Conversation"):
        for k in ['chat_session', 'messages']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# ── Gate: need API key ────────────────────────────────────────────────────────
if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to start chatting.", icon="🔑")
    st.stop()

# ── Initialise chat session ───────────────────────────────────────────────────
health_context = {
    'skin_result'  : st.session_state.get('skin_result'),
    'risk_result'  : st.session_state.get('risk_result'),
    'report_result': st.session_state.get('report_result'),
}

if 'chat_session' not in st.session_state:
    with st.spinner("Initialising MedBot…"):
        st.session_state.chat_session = create_chat_session(health_context, api_key)
    st.session_state.messages = []

if 'messages' not in st.session_state:
    st.session_state.messages = []

# ── Welcome message (first load) ──────────────────────────────────────────────
if not st.session_state.messages:
    welcome = (
        "👋 Hello! I'm **MedBot**, your AI health assistant.\n\n"
        "I can answer questions about your health based on your earlier module results. "
        "What would you like to know?"
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# ── Display conversation history ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask MedBot a health question…")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Emergency shortcut — no API call needed
    if is_emergency(user_input):
        from utils.helpers import get_emergency_response
        response = get_emergency_response()
    else:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("MedBot is thinking…"):
                response = send_chat_message(st.session_state.chat_session, user_input)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # Re-render so new messages show in the right order
    st.rerun()

# ── Suggested questions ───────────────────────────────────────────────────────
st.divider()
st.markdown("**💡 Suggested questions:**")
suggestions = [
    "What do my results mean overall?",
    "What diet changes should I make?",
    "Should I be worried about my lab results?",
    "How can I reduce my diabetes risk?",
    "What lifestyle changes do you recommend?",
]
cols = st.columns(len(suggestions))
for col, suggestion in zip(cols, suggestions):
    with col:
        if st.button(suggestion, use_container_width=True):
            # Inject as if user typed it
            st.session_state.messages.append({"role": "user", "content": suggestion})
            with st.spinner("MedBot is thinking…"):
                response = send_chat_message(st.session_state.chat_session, suggestion)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

st.divider()
st.warning(
    "⚠️ **MedBot is not a doctor.** Answers are for educational purposes only. "
    "Always consult a qualified healthcare professional for medical decisions.",
    icon="⚠️"
)
