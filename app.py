import streamlit as st
import joblib
import pandas as pd

if "predicted" not in st.session_state:
    st.session_state.predicted = False
# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
    }

    textarea, input {
        background-color: #1f2933 !important;
        color: white !important;
        border-radius: 8px;
    }

    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #ff6b6b;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("🧭 About This App")
st.sidebar.write(
    """
    This AI-powered system recommends suitable **career domains**
    based on your background, skills, and interests.

    ⚠️ This is a decision-support tool, not a definitive career verdict.
    """
)

# -------------------------
# Main Title
# -------------------------
st.title("Career Domain Recommendation System")
st.write("Enter your details to get a recommended career domain.")

# -------------------------
# Load Model
# -------------------------
model = joblib.load("models/career_domain_model.pkl")

# -------------------------
# Domain Descriptions
# -------------------------
domain_descriptions = {
    "Data & AI": "Data-driven roles involving analytics, machine learning, and AI.",
    "Software Engineering": "Roles focused on building and maintaining software systems.",
    "Design": "User experience, research, and visual design-focused roles.",
    "Business & Management": "Strategy, analytics, consulting, and management roles.",
    "Marketing": "Brand, content, and digital marketing roles.",
    "Cybersecurity": "Protecting systems, networks, and data from threats.",
    "Other": "Interdisciplinary or emerging roles."
}

# -------------------------
# User Inputs
# -------------------------
st.subheader("👤 Your Profile")

age = st.number_input("Age", min_value=18, max_value=60, value=22)

education = st.selectbox(
    "Highest Education Level",
    ["Bachelor's", "Master's", "PhD"]
)

skills = st.text_area(
    "Skills",
    placeholder="e.g. Python, Machine Learning, SQL, Power BI"
)

interests = st.text_area(
    "Interests",
    placeholder="e.g. Data Science, AI, Business Strategy"
)

st.caption(
    "💡 Recommendations are based on learned patterns from historical profiles "
    "and may vary for interdisciplinary backgrounds."
)
# -------------------------
# Prediction
# -------------------------
if st.button("Predict Career Domain"):
    st.session_state.predicted = True
    input_data = pd.DataFrame([{
        "Age": age,
        "Education": education,
        "Skills": skills,
        "Interests": interests
    }])

    probs = model.predict_proba(input_data)[0]
    classes = model.classes_

    top_indices = probs.argsort()[-2:][::-1]

    top_1 = classes[top_indices[0]]
    conf_1 = probs[top_indices[0]] * 100

    top_2 = classes[top_indices[1]]
    conf_2 = probs[top_indices[1]] * 100

    st.success(f"🎯 **Primary Recommendation:** {top_1} ({conf_1:.1f}%)")
    st.info(f"🔎 **Alternative Option:** {top_2} ({conf_2:.1f}%)")

    st.markdown(
        """
        ### 🧭 Understanding Your Career Recommendation
        - The **primary domain** best matches your current skills and background.
        - The **alternative option** highlights a closely related domain.
        - For interdisciplinary profiles, exploring both domains is beneficial.
        """
    )

    st.markdown("### 📌 Domain Overview")
    st.write(f"**{top_1}**")
    st.write(domain_descriptions.get(top_1))

    st.write("---")

    st.write(f"**Alternative: {top_2}**")
    st.write(domain_descriptions.get(top_2))

 # -------------------------
# Feedback (AFTER prediction)
# -------------------------
if st.session_state.predicted:
    st.markdown("### 📝 Feedback")

    feedback = st.radio(
        "Was this recommendation helpful?",
        ["Yes 👍", "Somewhat?", "No 👎"],
        key="feedback_radio"
    )

    comments = st.text_area(
        "Optional: Tell us how we can improve",
        placeholder="Your feedback helps improve future recommendations...",
        key="feedback_comments"
    )

    if st.button("Submit Feedback", key="submit_feedback"):
        st.success("Thank you for your feedback! 💙")
   

# ----
