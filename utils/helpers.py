"""
utils/helpers.py
Shared utility functions — model loading, inference, Gemini calls.
All functions used across the 5 Streamlit pages live here.
"""

import os
import re
import json
import math
import tempfile
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Optional heavy imports (caught if not installed) ─────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    import tensorflow.keras.backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Google Drive model ID ─────────────────────────────────────────────────────
import gdown
SKIN_MODEL_GDRIVE_ID = "13WjfBkD8wbS0XVfDqj9S53H8vM-X6I3I"
MODEL_FILENAME = "skin_weights_only.weights.h5"

def download_model_if_needed(model_path: str, gdrive_id: str):
    import os

    if os.path.exists(model_path):
        print("✅ Model already exists at:", model_path)
        return True

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        print("⬇️ Downloading model from:", url)

        import gdown
        gdown.download(url, model_path, quiet=False, fuzzy=True)

        print("📂 Exists after download:", os.path.exists(model_path))

        return os.path.exists(model_path)

    except Exception as e:
        print(f"[Download error] {e}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — SKIN DISEASE
# ═══════════════════════════════════════════════════════════════════════════════

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss — needed to load the saved .keras model."""
    def focal_loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        p_t = K.sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = K.pow(1.0 - p_t, gamma)
        focal_ce = alpha * focal_weight * ce
        return K.mean(K.sum(focal_ce, axis=-1))
    focal_loss_fn.__name__ = 'focal_loss_fn'
    return focal_loss_fn



@st.cache_resource
def load_skin_model(model_path: str, names_path: str):
    model_path = os.path.join(BASE_DIR, model_path) if not os.path.isabs(model_path) else model_path
    names_path = os.path.join(BASE_DIR, names_path) if not os.path.isabs(names_path) else names_path

    download_model_if_needed(model_path, SKIN_MODEL_GDRIVE_ID)

    if not TF_AVAILABLE:
        return None, None

    try:
        with open(names_path, 'rb') as f:
            class_names = pickle.load(f)
        num_classes = len(class_names)
        print(f"Class names loaded: {num_classes} classes")

        # Rebuild architecture — pure Python, no config file
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        from tensorflow.keras import layers as L

        base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
        base.trainable = False

        inp  = tf.keras.Input(shape=(224,224,3))
        x    = base(inp, training=False)
        x    = L.GlobalAveragePooling2D()(x)
        x    = L.BatchNormalization()(x)
        x    = L.Dense(512, activation='relu')(x)
        x    = L.Dropout(0.4)(x)
        x    = L.Dense(256, activation='relu')(x)
        x    = L.Dropout(0.3)(x)
        out  = L.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inp, out)

        # Load weights — works on any Keras version, no config parsing
        model.load_weights(model_path)
        print("✅ Weights loaded successfully")

        # Quick sanity check
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        pred  = model.predict(dummy, verbose=0)
        print(f"✅ Model working. Output shape: {pred.shape}")

        return model, class_names

    except Exception as e:
        st.error("❌ Skin Model Load Error")
        st.code(traceback.format_exc())
        return None, None

def predict_skin_disease(image_input, model, class_names, top_k: int = 3):
    """
    Predict skin disease from a PIL Image or file path.

    Args:
        image_input  : PIL.Image or str path
        model        : loaded Keras model
        class_names  : list of class name strings
        top_k        : number of top predictions to return

    Returns:
        list of (class_name, confidence_pct) tuples
    """
    if not TF_AVAILABLE or not PIL_AVAILABLE:
        return []

    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)              # MobileNetV2 normalisation → [-1, 1]
    arr = np.expand_dims(arr, axis=0)        # (224,224,3) → (1,224,224,3)

    probs   = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], round(float(probs[i]) * 100, 2)) for i in top_idx]


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — HEALTH RISK
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FEATURES = [
    'HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
    'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
    'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
    'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'
]


@st.cache_resource   
def load_risk_model(model_path: str, features_path: str):

    # Fix paths
    model_path = os.path.join(BASE_DIR, model_path) if not os.path.isabs(model_path) else model_path
    features_path = os.path.join(BASE_DIR, features_path) if not os.path.isabs(features_path) else features_path

     # DEBUG
    print("MODEL PATH:", model_path)
    print("FEATURE PATH:", features_path)
    print("MODEL EXISTS:", os.path.exists(model_path))
    print("FEATURE EXISTS:", os.path.exists(features_path))

    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features

    except Exception as e:
        print(f"[Risk model load error] {e}")
        return None, None

def predict_diabetes_risk(input_data: dict, model, features: list) -> dict:
    """
    Predict diabetes risk from a dict of health indicators.

    Returns:
        dict with risk_level, probability, top_factors
    """
    row  = pd.DataFrame([input_data])[features]
    prob = model.predict_proba(row)[0][1]

    if prob < 0.30:
        risk_level = "Low"
        color      = "green"
    elif prob < 0.60:
        risk_level = "Moderate"
        color      = "orange"
    else:
        risk_level = "High"
        color      = "red"

    # SHAP top factors
    top_factors = []
    if SHAP_AVAILABLE:
        try:
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer.shap_values(row)[0]
            feat_shap   = list(zip(features, shap_vals))
            feat_shap.sort(key=lambda x: abs(x[1]), reverse=True)
            top_factors = [(f, round(float(v), 4)) for f, v in feat_shap[:5]]
        except Exception:
            pass

    return {
        'risk_level' : risk_level,
        'probability': round(float(prob) * 100, 2),
        'color'      : color,
        'top_factors': top_factors,
    }


def make_shap_waterfall_fig(input_data: dict, model, features: list):
    """Return a matplotlib Figure of the SHAP waterfall for one patient."""
    if not SHAP_AVAILABLE:
        return None
    try:
        row       = pd.DataFrame([input_data])[features]
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(row)
        explanation = shap.Explanation(
            values        = shap_vals[0],
            base_values   = explainer.expected_value,
            data          = row.iloc[0].values,
            feature_names = features
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(explanation, max_display=10, show=False)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — REPORT EXPLAINER
# ═══════════════════════════════════════════════════════════════════════════════

TESS_CONFIG = '--psm 4 --oem 3 -l eng'

REPORT_SYSTEM_PROMPT = """You are a compassionate, expert medical report interpreter.
Your job is to help patients understand their medical lab reports in simple, clear language.

When given extracted text from a lab report, you MUST:
1. Identify all lab test names, values, units, and reference ranges
2. For each test: explain what it measures, what the result means, and whether it is Normal, Low, or High
3. Provide a brief overall health summary in plain, patient-friendly English
4. List any values that need immediate medical attention — flag these clearly
5. Give 3-5 actionable next steps the patient can take
6. ALWAYS include this exact disclaimer at the end

Format your response as valid JSON with EXACTLY these keys:
{
  "patient_summary": "one paragraph plain-language overview",
  "test_results": [
    {
      "test_name": "name of the test",
      "value": "result value",
      "unit": "unit of measurement",
      "reference_range": "normal range",
      "status": "Normal or Low or High or Critical",
      "explanation": "what this test measures",
      "patient_friendly": "what this result means in simple language"
    }
  ],
  "flags": ["list any critical or concerning values here"],
  "next_steps": ["list 3-5 specific recommended actions"],
  "disclaimer": "This analysis is for educational purposes only. It is NOT a medical diagnosis. Always consult a qualified healthcare professional before making any health decisions."
}
"""


def preprocess_for_ocr(pil_image):
    """
    5-step OCR preprocessing on a PIL Image.
    Returns a preprocessed np.ndarray (binary image).
    """
    if not CV2_AVAILABLE:
        return np.array(pil_image.convert('L'))

    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised  = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(denoised)
    binary    = cv2.adaptiveThreshold(enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, blockSize=11, C=2)

    # Deskew
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) > 100:
        rect  = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            h, w = binary.shape
            M    = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binary = cv2.warpAffine(binary, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return binary


def extract_ocr_text(pil_image) -> str:
    """Run Tesseract on a preprocessed PIL image. Returns cleaned text string."""
    if not OCR_AVAILABLE or not PIL_AVAILABLE:
        return ""
    processed   = preprocess_for_ocr(pil_image)
    pil_bin     = Image.fromarray(processed)
    raw_text    = pytesseract.image_to_string(pil_bin, config=TESS_CONFIG)
    # Clean
    text = re.sub(r'[^\x20-\x7E\n]', ' ', raw_text)
    lines = [l.strip() for l in text.split('\n')
             if len(l.strip()) >= 3 and any(c.isalnum() for c in l)]
    text = '\n'.join(lines)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def explain_report_gemini(ocr_text: str, patient_age=None, patient_gender=None,
                           api_key: str = None) -> dict:
    if not GEMINI_AVAILABLE or not api_key:
        return {'parse_error': True, 'raw_response': 'Gemini not configured.'}

    client = genai.Client(api_key=api_key)

    context = ""
    if patient_age:    context += f" Patient age: {patient_age}."
    if patient_gender: context += f" Patient gender: {patient_gender}."

    prompt = (f"{REPORT_SYSTEM_PROMPT}\n\n"
              f"Please analyse and explain this medical lab report.{context}\n\n"
              f"--- REPORT TEXT START ---\n{ocr_text}\n--- REPORT TEXT END ---\n\n"
              f"Return ONLY valid JSON.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.2)
        )
        result_text = response.text.strip()
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            result_text = match.group()
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {'parse_error': True, 'raw_response': result_text}
    except Exception as e:
        return {'parse_error': True, 'raw_response': str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

EMERGENCY_KEYWORDS = [
    'chest pain', 'cannot breathe', "can't breathe", 'heart attack',
    'suicidal', 'want to die', 'collapse', 'unconscious', 'stroke',
    'seizure', 'overdose', 'bleeding heavily'
]

CHATBOT_SYSTEM_TEMPLATE = """You are MedBot, a compassionate and knowledgeable AI health assistant.
You are part of MedAI Nexus, an AI-powered healthcare assistant system.

ABOUT THIS USER (from previous modules):
- Skin disease result: {skin}
- Diabetes risk assessment: {risk}
- Lab report findings: {report}

YOUR RULES — follow strictly:
1. PERSONALISE every response using the user's data above.
2. NEVER diagnose any condition. Say 'Your assessment suggests...'
3. ALWAYS recommend consulting a qualified doctor for medical decisions.
4. EMERGENCY: If user mentions chest pain, breathing difficulty, suicidal thoughts —
   respond: "⚠️ Please contact emergency services (112 / 108 in India) immediately."
5. STAY within health-related topics only.
6. TONE: Warm, supportive, simple English.
7. RESPONSE LENGTH: 3–5 sentences unless more detail is needed.
"""


def build_chatbot_system_prompt(health_context: dict) -> str:
    return CHATBOT_SYSTEM_TEMPLATE.format(
        skin   = health_context.get('skin_result')   or 'Not assessed yet',
        risk   = health_context.get('risk_result')   or 'Not assessed yet',
        report = health_context.get('report_result') or 'Not assessed yet',
    )


def is_emergency(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in EMERGENCY_KEYWORDS)


def get_emergency_response() -> str:
    return (
        "⚠️ **URGENT:** Based on what you described, please **call emergency services immediately.**\n\n"
        "🚨 **India:** Call **112** (national) or **108** (ambulance)\n\n"
        "Please stop and seek help right now. Do not delay."
    )


def create_chat_session(health_context: dict, api_key: str):
    if not GEMINI_AVAILABLE or not api_key:
        return None, None
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=genai_types.GenerateContentConfig(
            system_instruction=build_chatbot_system_prompt(health_context)
        )
    )
    return client, chat   # return BOTH — client must stay alive


def send_chat_message(chat_session, user_message: str) -> str:
    if chat_session is None:
        return "MedBot is not configured. Please add your Gemini API key in the sidebar."
    if is_emergency(user_message):
        return get_emergency_response()
    try:
        response = chat_session.send_message(user_message)
        return response.text
    except Exception as e:
        return f"Sorry, I had trouble responding. Please try again. ({str(e)[:80]})"

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — LIFESTYLE COACH
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lifestyle_scores(data: dict) -> dict:
    """Compute 0-100 scores for 6 lifestyle dimensions + overall."""
    scores = {}

    hours = data.get('sleep_hours', 7)
    scores['Sleep'] = 100 if 7 <= hours <= 9 else max(0, 100 - abs(hours - 8) * 15)

    intensity_map = {'None': 0, 'Light': 40, 'Moderate': 75, 'Intense': 100}
    intensity     = intensity_map.get(data.get('activity_level', 'None'), 0)
    days          = data.get('active_days_per_week', 0)
    scores['Activity'] = min(100, intensity * 0.5 + (days / 7) * 50)

    fruit_pts = min(data.get('fruit_servings', 0), 5) * 4
    veg_pts   = min(data.get('veg_servings', 0),   5) * 4
    water_pts = min(data.get('water_litres', 0) * 16, 40)
    diet_bonus = {'Vegetarian': 10, 'Vegan': 15, 'Mixed': 5, 'Carnivore': 0}
    scores['Nutrition'] = min(100, fruit_pts + veg_pts + water_pts +
                              diet_bonus.get(data.get('diet_type', 'Mixed'), 5))

    habit = 100
    if data.get('smoker', False): habit -= 40
    habit -= min(data.get('alcohol_units_per_week', 0) * 3, 40)
    scores['Habits'] = max(0, habit)

    stress = data.get('stress_level', 5)
    scores['Mental'] = max(0, 100 - (stress - 1) * 11)
    quality_bonus = {'Poor': 0, 'Fair': 3, 'Good': 7, 'Excellent': 10}
    scores['Mental'] = min(100, scores['Mental'] +
                           quality_bonus.get(data.get('sleep_quality', 'Fair'), 3))

    screen = data.get('screen_hours_per_day', 0)
    scores['Screen'] = max(0, 100 - max(0, screen - 4) * 12)

    scores['Overall'] = round(sum(v for k, v in scores.items()) / 6, 1)
    return scores


def make_radar_fig(scores: dict) -> plt.Figure:
    """Return a matplotlib Figure of the lifestyle radar chart."""
    dims   = ['Sleep', 'Activity', 'Nutrition', 'Habits', 'Mental', 'Screen']
    values = [scores.get(d, 0) for d in dims]
    n      = len(dims)
    angles = [2 * math.pi * i / n for i in range(n)]

    vals_plot   = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, vals_plot, 'o-', linewidth=2, color='#1A56A0', markersize=4)
    ax.fill(angles_plot, vals_plot, alpha=0.2, color='#1A56A0')
    ax.plot(angles_plot, [70] * (n + 1), '--', color='red', alpha=0.6,
            linewidth=1.5, label='Target (70)')
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{d}\n{scores.get(d, 0):.0f}" for d in dims], fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_title(f"Lifestyle Scores\nOverall: {scores.get('Overall', 0):.0f}/100",
                 fontsize=11, fontweight='bold', pad=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    return fig


WELLNESS_SYSTEM_PROMPT = """You are an expert wellness coach and nutritionist.
You create personalised, practical, motivating 7-day wellness improvement plans.
Return ONLY valid JSON with this exact structure:
{
  "health_score_summary": "2-3 sentences",
  "priority_areas": ["area1", "area2"],
  "weekly_plan": {
    "Monday":    {"morning": "...", "afternoon": "...", "evening": "..."},
    "Tuesday":   {"morning": "...", "afternoon": "...", "evening": "..."},
    "Wednesday": {"morning": "...", "afternoon": "...", "evening": "..."},
    "Thursday":  {"morning": "...", "afternoon": "...", "evening": "..."},
    "Friday":    {"morning": "...", "afternoon": "...", "evening": "..."},
    "Saturday":  {"morning": "...", "afternoon": "...", "evening": "..."},
    "Sunday":    {"morning": "...", "afternoon": "...", "evening": "..."}
  },
  "nutrition_tips": ["tip1", "tip2", "tip3"],
  "mental_wellness_tips": ["tip1", "tip2", "tip3"],
  "specific_recommendations": "personalised advice based on health results",
  "motivational_message": "one encouraging message",
  "disclaimer": "This plan is for general wellness guidance only. Consult a healthcare professional before significant changes."
}"""


def generate_wellness_plan(lifestyle_data: dict, scores: dict,
                            health_context: dict, api_key: str) -> dict:
    if not GEMINI_AVAILABLE or not api_key:
        return {'parse_error': True, 'raw_response': 'Gemini not configured.'}

    client = genai.Client(api_key=api_key)

    # ── Build prompt (THIS IS MISSING) ─────────────────────────
    parts = []

    parts.append(WELLNESS_SYSTEM_PROMPT)
    parts.append("\nSTRICT RULE: Fill ALL 7 days (Monday–Sunday) with morning, afternoon, evening. Do NOT leave empty."))

    parts.append("\nUSER LIFESTYLE SCORES:")
    for dim, score in scores.items():
        if dim != 'Overall':
           parts.append(f"{dim}: {score}/100")
    parts.append(f"Overall: {scores.get('Overall', 0)}/100")

    parts.append("\nUSER LIFESTYLE DETAILS:")
    for k, v in lifestyle_data.items():
        parts.append(f"{k}: {v}")

    if health_context:
        parts.append("\nPREVIOUS HEALTH RESULTS:")
        for k, v in health_context.items():
            if v and v != 'Not assessed yet':
                parts.append(f"{k}: {v}")

    prompt = "\n".join(parts)                            
   

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.6,
                max_output_tokens=3000,
                response_mime_type="application/json"    
            )
        )
        result_text = response.text.strip()
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            result_text = match.group()
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {'parse_error': True, 'raw_response': result_text}
    except Exception as e:
        return {'parse_error': True, 'raw_response': str(e)}
