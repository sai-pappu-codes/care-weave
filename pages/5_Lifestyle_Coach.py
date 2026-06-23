import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.helpers import compute_lifestyle_scores, make_radar_fig, generate_wellness_plan

st.set_page_config(page_title="Lifestyle Coach | MedAI Nexus", page_icon="🌿", layout="wide")

st.title("🌿 Module 5 — Lifestyle Coach")
st.caption("Fill in your lifestyle habits. Get a scored assessment and personalised 7-day wellness plan.")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input("Enter your Gemini API key", type="password",
                             key="coach_api_key",
                             help="Free at: https://aistudio.google.com/app/apikey")
    st.caption("Required for the 7-day wellness plan.")
    st.divider()
    st.markdown("### Health Context")
    st.markdown(f"🔬 {st.session_state.get('skin_result','Not assessed')[:40]}")
    st.markdown(f"💉 {st.session_state.get('risk_result','Not assessed')[:40]}")
    st.markdown(f"📋 {st.session_state.get('report_result','Not assessed')[:40]}")

# ── Lifestyle form ────────────────────────────────────────────────────────────
with st.form("lifestyle_form"):
    st.markdown("### Your Daily Lifestyle")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**😴 Sleep**")
        sleep_hours   = st.slider("Sleep hours per night", 3.0, 12.0, 7.0, 0.5)
        sleep_quality = st.selectbox("Sleep quality", ["Poor","Fair","Good","Excellent"], index=1)

        st.markdown("**🏃 Activity**")
        activity_level     = st.selectbox("Exercise intensity", ["None","Light","Moderate","Intense"], index=1)
        active_days        = st.slider("Active days per week", 0, 7, 3)

    with col2:
        st.markdown("**🥗 Nutrition**")
        fruit_servings = st.slider("Fruit servings/day", 0, 10, 2)
        veg_servings   = st.slider("Vegetable servings/day", 0, 10, 2)
        water_litres   = st.slider("Water intake (litres/day)", 0.0, 5.0, 2.0, 0.25)
        diet_type      = st.selectbox("Diet type", ["Mixed","Vegetarian","Vegan","Carnivore"])

    with col3:
        st.markdown("**🚬 Habits**")
        smoker      = st.radio("Do you smoke?", [False, True],
                               format_func=lambda x: "No" if not x else "Yes", horizontal=True)
        alcohol_units = st.slider("Alcohol units/week", 0, 30, 2)

        st.markdown("**🧠 Mental & Screen**")
        stress_level  = st.slider("Stress level (1=low, 10=high)", 1, 10, 5)
        screen_hours  = st.slider("Daily screen time (hrs, non-work)", 0, 16, 4)

    submitted = st.form_submit_button("📊 Analyse My Lifestyle", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if submitted:
    lifestyle_data = {
        'sleep_hours'           : sleep_hours,
        'sleep_quality'         : sleep_quality,
        'activity_level'        : activity_level,
        'active_days_per_week'  : active_days,
        'fruit_servings'        : fruit_servings,
        'veg_servings'          : veg_servings,
        'water_litres'          : water_litres,
        'diet_type'             : diet_type,
        'smoker'                : smoker,
        'alcohol_units_per_week': alcohol_units,
        'stress_level'          : stress_level,
        'screen_hours_per_day'  : screen_hours,
    }

    scores = compute_lifestyle_scores(lifestyle_data)
    st.session_state['lifestyle_scores']  = scores
    st.session_state['lifestyle_data']    = lifestyle_data

    st.divider()
    st.markdown("## Your Lifestyle Assessment")

    # Overall badge
    overall = scores['Overall']
    if overall >= 70:
        st.success(f"🌟 Overall Score: **{overall:.0f}/100** — Good lifestyle! Keep it up.")
    elif overall >= 50:
        st.warning(f"⚡ Overall Score: **{overall:.0f}/100** — Room for improvement.")
    else:
        st.error(f"⚠️ Overall Score: **{overall:.0f}/100** — Several areas need attention.")

    # Score breakdown + radar chart
    col_scores, col_radar = st.columns([1, 1])

    with col_scores:
        st.markdown("#### Score Breakdown")
        dims = ['Sleep','Activity','Nutrition','Habits','Mental','Screen']
        for dim in dims:
            score  = scores[dim]
            bar    = int(score / 5)
            status = "Good" if score >= 70 else "Fair" if score >= 50 else "Needs work"
            color  = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            st.markdown(
                f"{color} **{dim}**: {score:.0f}/100 — {status}  \n"
                f"`{'█' * bar}{'░' * (20-bar)}`"
            )
            st.write("")

    with col_radar:
        fig = make_radar_fig(scores)
        st.pyplot(fig, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close()

    # Priority areas
    dim_scores  = {d: scores[d] for d in dims}
    bottom3     = sorted(dim_scores, key=dim_scores.get)[:3]
    st.markdown(f"#### 🎯 Focus Areas: **{', '.join(bottom3)}**")

    # Generate Gemini plan
    st.divider()
    st.markdown("## Your 7-Day Personalised Wellness Plan")

    if not api_key:
        st.info("👈 Add your Gemini API key in the sidebar to generate a personalised 7-day plan.", icon="🔑")
    else:
        health_context = {
            'skin_result'   : st.session_state.get('skin_result'),
            'risk_result'   : st.session_state.get('risk_result'),
            'report_result' : st.session_state.get('report_result'),
        }

        with st.spinner("Gemini is writing your personalised 7-day wellness plan (10-20 seconds)…"):
            plan = generate_wellness_plan(lifestyle_data, scores, health_context, api_key)

        if plan.get('parse_error'):
            st.error("Could not generate plan. Try again.")
            with st.expander("Debug info"):
                st.code(plan.get('raw_response','')[:500])
        else:
            # Summary and priority
            if plan.get('health_score_summary'):
                st.info(f"📊 **Summary:** {plan['health_score_summary']}")
            if plan.get('priority_areas'):
                st.markdown(f"🎯 **Priority Areas:** {', '.join(plan['priority_areas'])}")

            # 7-day plan
            st.markdown("### 📅 Weekly Plan")
            days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            weekly = plan.get('weekly_plan', {})

           if not weekly or not isinstance(weekly, dict):
               st.error("⚠️ Invalid plan format. Please regenerate.")
               st.stop())
               
            tabs   = st.tabs(days)
            for tab, day in zip(tabs, days):
                with tab:
                    day_plan = weekly.get(day, {})
                    st.markdown(f"**🌅 Morning:** {day_plan.get('morning','—')}")
                    st.markdown(f"**☀️ Afternoon:** {day_plan.get('afternoon','—')}")
                    st.markdown(f"**🌙 Evening:** {day_plan.get('evening','—')}")

            # Tips in two columns
            col_nut, col_ment = st.columns(2)
            with col_nut:
                if plan.get('nutrition_tips'):
                    st.markdown("### 🥗 Nutrition Tips")
                    for tip in plan['nutrition_tips']:
                        st.markdown(f"- {tip}")
            with col_ment:
                if plan.get('mental_wellness_tips'):
                    st.markdown("### 🧘 Mental Wellness Tips")
                    for tip in plan['mental_wellness_tips']:
                        st.markdown(f"- {tip}")

            # Personalised recommendation
            if plan.get('specific_recommendations'):
                st.markdown("### 💡 Personalised Advice")
                st.markdown(plan['specific_recommendations'])

            # Motivational message
            if plan.get('motivational_message'):
                st.success(f"✨ {plan['motivational_message']}")

            # Disclaimer
            st.divider()
            st.warning(plan.get('disclaimer',
                "⚠️ This plan is for general wellness guidance only. "
                "Consult a healthcare professional before significant changes."), icon="⚠️")
