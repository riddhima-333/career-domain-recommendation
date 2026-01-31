# Career Domain Recommendation System

An end-to-end machine learning application that recommends suitable **career domains**
based on a user’s background, skills, and interests.  
The system is designed as a **decision-support tool**, not a definitive career predictor.

🔗 **Live App:** https://riddhima-333-career-domain-recommendation-app-gsowdu.streamlit.app/

---

## Problem Statement

Career guidance systems often attempt to predict **specific job titles**, which can be:
- overly rigid
- noisy with limited data
- misleading for interdisciplinary profiles

This project reframes the problem as a **career domain recommendation task**, providing
more flexible, interpretable, and realistic guidance.

---

## Approach & Methodology

### 1️⃣ Data Processing
- Used a structured dataset containing:
  - Age (numerical)
  - Education (categorical)
  - Skills & Interests (textual)
- Converted unstructured text using **TF-IDF vectorization**
- Encoded categorical variables using **One-Hot Encoding**

### 2️⃣ Model Selection
- Implemented a **Random Forest Classifier** for:
  - robustness on small datasets
  - reduced overfitting compared to single decision trees
- Chose domain-level prediction instead of job-level labels due to:
  - class imbalance
  - limited samples for niche roles

### 3️⃣ Handling Imbalanced Classes
- Underrepresented domains were merged into an **“Other”** category
- This improved model stability and interpretability

### Interpretability
- The system outputs:
  - **Primary recommended domain**
  - **Alternative domain**
  - **Confidence scores**
- Added domain descriptions and interpretation guidance for clarity

---

## Application Features

- Interactive **Streamlit UI**
- Domain-level career recommendation
- Confidence-based Top-2 predictions
- Domain explanations
- User feedback collection
- Clean, responsive interface with custom styling

---

## Tech Stack

- **Python**
- **scikit-learn**
- **pandas**
- **NumPy**
- **Streamlit**
- **joblib**

---

## Deployment

- Deployed using **Streamlit Cloud**
- Model and environment versions are pinned for compatibility
- CI-friendly GitHub-based deployment workflow

---

## Disclaimer

This system is intended for **educational and advisory purposes only**.  
Career decisions should consider multiple factors and professional guidance.

---

## Future Improvements

- Collect larger and more diverse datasets
- Multi-label domain recommendations
- Skill-gap analysis and upskilling suggestions
- Integration of contextual embeddings (e.g., BERT)
- Feedback-driven model retraining

---

## Author

**Riddhima Dutta**

