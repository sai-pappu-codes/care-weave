# 🏥 MedAI Nexus 
AI-Powered Multi-Modal Healthcare Assistant
---

> **Capstone Project** | Data Science with GenAI | Entri Elevate × NSDC × Illinois Tech  
> **Built by:** Fathima Salga | Kochi, Kerala  
> **Live App:** [https://pfveklr9pexfstzg65gq8s.streamlit.app](https://pfveklr9pexfstzg65gq8s.streamlit.app)

---

##  About the Project

MedAI Nexus is a **5-module AI-powered healthcare assistant** built and deployed end-to-end as a capstone project. It helps patients understand their own health data through computer vision, machine learning, OCR, and large language models — all integrated into a single Streamlit web application.

> **Medical Disclaimer:** This is a student project for educational purposes only. All outputs are NOT medical diagnoses. Always consult a qualified healthcare professional before making any health decisions.

---

##  Live Demo

 **[Open MedAI Nexus](https://pfveklr9pexfstzg65gq8s.streamlit.app)**

| Module | What it does |
|--------|-------------|
| 🔬 Skin Disease | Upload a skin photo → 20-class AI classification |
| 💉 Health Risk | Fill health form → Diabetes risk + SHAP explanation |
| 📋 Report Explainer | Upload lab report → OCR + Gemini plain-language explanation |
| 💬 AI Chatbot | Ask health questions → Personalised answers using your results |
| 🌿 Lifestyle Coach | Fill habits form → 6-dimension score + 7-day wellness plan |

---

##  System Architecture

```
User uploads skin image  →  Module 1: MobileNetV2  →  skin_result
User fills health form   →  Module 2: XGBoost      →  risk_result
User uploads lab report  →  Module 3: OCR + Gemini →  report_result
                                      ↓
                         st.session_state (shared)
                                      ↓
                    Module 4: AI Chatbot  (uses all 3 results)
                    Module 5: Lifestyle Coach (uses all 3 results)
```

All five modules share data through Streamlit's `session_state` — results from earlier modules automatically personalise later ones.

---

##  Modules — Technical Details

### Module 1 — Skin Disease Prediction

| Detail | Value |
|--------|-------|
| Model | MobileNetV2 (Transfer Learning) |
| Dataset | DermNet — 19,500+ images, 20 classes |
| Imbalance | 6.6:1 ratio → solved with Focal Loss (γ=2.0) |
| Accuracy | 41.6% (20-class, random baseline = 5%) |
| Macro F1 | 0.368 |
| Training | 2 phases — frozen base → fine-tune top 40 layers |

**Key challenge:** Model saved with TF 2.19 (Keras 3.5+) had `quantization_config` in Dense layer config — caused deserialization failure on deployment. **Fix:** Rebuilt architecture in pure Python + `save_weights()` to `.weights.h5` (12.1MB) — bypasses config entirely.

---

### Module 2 — Health Risk Prediction

| Detail | Value |
|--------|-------|
| Model | XGBoost + Random Forest |
| Dataset | CDC BRFSS 2015 — 253,680 rows, 21 features |
| Imbalance | 4.78:1 ratio → solved with SMOTE only |
| XGBoost Accuracy | 83.7% |
| XGBoost AUC-ROC | 0.813 |
| RF Diabetic Recall | 50% (better for screening) |
| Explainability | SHAP TreeExplainer + waterfall plot |

**Key challenge:** Using SMOTE + `scale_pos_weight` simultaneously caused model collapse (logloss 0.691 → 0.746). Fix: Remove `scale_pos_weight`, keep SMOTE only.

> **Note:** Random Forest chosen for the app despite lower accuracy because it achieves 50% diabetic recall vs XGBoost's 22% — recall matters more than accuracy in medical screening.

---

###  Module 3 — Medical Report Explainer

| Detail | Value |
|--------|-------|
| OCR Engine | Tesseract (--psm 4 --oem 3) |
| LLM | Gemini 2.5 Flash |
| Preprocessing | 5-step pipeline |
| OCR Accuracy | ~40% raw → ~85% after preprocessing |
| Output | Structured JSON — patient_summary, test_results, flags, next_steps |

**5-Step OCR Preprocessing Pipeline:**
```
Raw Image → Grayscale → Denoise → CLAHE → Binarise → Deskew → Tesseract
```

---

###  Module 4 — AI Health Chatbot

| Detail | Value |
|--------|-------|
| LLM | Gemini 2.5 Flash |
| Memory | `client.chats.create()` — auto history management |
| Personalisation | System instruction includes Module 1-3 results |
| Safety | 12 emergency keywords detected before API call |

---

###  Module 5 — Lifestyle Coach

| Detail | Value |
|--------|-------|
| LLM | Gemini 2.5 Flash |
| Scoring | 6 dimensions, 0-100 mathematical formulas |
| Visualisation | Matplotlib radar chart |
| Output | Full 7-day wellness plan (JSON) |

**6 Scoring Dimensions:**

| Dimension | Formula |
|-----------|---------|
| Sleep | 100 if 7-9 hrs, else -15 per hour off |
| Activity | Intensity × 0.5 + (active days/7) × 50 |
| Nutrition | Fruit + Veg + Water + Diet bonus |
| Habits | 100 - 40(smoke) - 3×(alcohol units) |
| Mental | 100 - (stress-1) × 11 + quality bonus |
| Screen | 100 - max(0, hours-4) × 12 |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| Deep Learning | TensorFlow 2.16.2 · Keras |
| Machine Learning | XGBoost 1.7.6 · Scikit-learn · SHAP |
| OCR | Tesseract 5.5 · pytesseract · OpenCV |
| LLM | Google Gemini 2.5 Flash (google-genai) |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Web App | Streamlit 1.32 |
| Model Storage | Google Drive + gdown |
| Deployment | Streamlit Cloud |
| Version Control | Git · GitHub |

---

## 📁 Project Structure

```
medai-nexus/
├── app.py                          ← Home page / entry point
├── requirements.txt                ← Python dependencies
├── packages.txt                    ← System packages (Tesseract)
├── pages/
│   ├── 1_Skin_Disease.py
│   ├── 2_Health_Risk.py
│   ├── 3_Report_Explainer.py
│   ├── 4_AI_Chatbot.py
│   └── 5_Lifestyle_Coach.py
├── utils/
│   └── helpers.py                  ← All inference functions
└── models/
    ├── skin_weights_only.weights.h5     ← Downloaded from Google Drive
    ├── skin_disease_class_names.pkl
    ├── health_risk_xgb.pkl
    └── health_features.pkl
```

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Tesseract OCR installed

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# Mac
brew install tesseract
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/fathimasalga/medai-nexus.git
cd medai-nexus

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Gemini API Key
Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and enter it in the sidebar when using Modules 3, 4, or 5.

---

##  Model Files

| File | Size | Location |
|------|------|----------|
| skin_weights_only.weights.h5 | 12.1 MB | Google Drive (auto-downloaded) |
| skin_disease_class_names.pkl | < 1 MB | GitHub repo |
| health_risk_xgb.pkl | ~5 MB | GitHub repo |
| health_features.pkl | < 1 MB | GitHub repo |

> The skin model weights file is hosted on Google Drive and downloaded automatically at first launch using `gdown`. It does not need to be uploaded to GitHub.

---

## Key Technical Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| LLM | Gemini 2.5 Flash | Free API — GPT-4o requires billing |
| Imbalance (Module 1) | Focal Loss | 6.6× imbalance — down-weights easy examples |
| Imbalance (Module 2) | SMOTE only | scale_pos_weight + SMOTE caused model collapse |
| Model saving | save_weights() .h5 | Avoids Keras version config conflicts |
| Model hosting | Google Drive + gdown | GitHub 25MB limit — weights file is 12.1MB |
| Evaluation metric | Recall over Accuracy | Medical screening — missing a case is worse than false alarm |

---

##  Challenges Solved

| # | Challenge | Solution |
|---|-----------|----------|
| 1 | 6.6× class imbalance in DermNet | Focal Loss (γ=2.0, α=0.25) |
| 2 | SMOTE + scale_pos_weight → model collapse | Removed scale_pos_weight, SMOTE only |
| 3 | Keras quantization_config error on deployment | Rebuilt architecture in Python + save_weights() |
| 4 | GitHub 25MB file size limit | Google Drive + gdown auto-download |
| 5 | GPT-4o billing required | Switched to Gemini 1.5 Flash → 2.5 Flash |
| 6 | google-generativeai SDK deprecated (Nov 2025) | Migrated to new google-genai SDK |
| 7 | Gemini chat client closing mid-session | Stored client in st.session_state |
| 8 | OCR accuracy ~40% on raw images | 5-step preprocessing pipeline → ~85% |
| 9 | Free GPU limits hit mid-training | Colab Pro + A100 GPU (still took 2 hours) |
| 10 | API quota exhaustion | User-provided keys + model rotation |

---

## Results Summary

| Module | Metric | Value |
|--------|--------|-------|
| Skin Disease | Overall Accuracy | 41.6% |
| Skin Disease | Macro F1-Score | 0.368 |
| Skin Disease | Best class (Nail Fungus) | 77% |
| Health Risk | XGBoost Accuracy | 83.7% |
| Health Risk | XGBoost AUC-ROC | 0.813 |
| Health Risk | RF Diabetic Recall | 50% |
| Report Explainer | OCR accuracy (after preprocessing) | ~85% |
| Lifestyle Coach | Scoring dimensions | 6 (0-100 each) |

---

##  About the Developer

**Fathima Salga**  
Aspiring Data Scientist | Kochi, Kerala

-  M.Sc Physical Oceanography — CUSAT (CGPA 8.29)
-  Ex-Project Associate — DRDO NPOL (2022-2024)
-  Data Science with GenAI — Entri × Illinois Tech × NSDC
-  [LinkedIn](https://www.linkedin.com/in/fathima-salga-4193ab20b/)
-  [GitHub](https://github.com/fathimasalga)

---

## License

This project is built for educational purposes as part of the Data Science with GenAI capstone program.  
All health outputs include a disclaimer: **Not a medical diagnosis. Always consult a qualified healthcare professional.**

---

<div align="center">
  <strong>MedAI Nexus</strong> · Built by Fathima Salga · Kochi, Kerala<br>
  TensorFlow · XGBoost · Tesseract · Gemini 2.5 Flash · Streamlit
</div>
