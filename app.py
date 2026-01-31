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
    /* Import cyber / futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');

    /* App background */
    .stApp {
        background: radial-gradient(circle at top, #1a2a6c, #000000);
        color: white;
    }

    /* Main content container */
    .block-container {
        border: 1px solid #00f2ff;
        box-shadow: 0 0 20px #00f2ff55;
        border-radius: 16px;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.6);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.75);
        border-right: 1px solid #00f2ff55;
    }

    /* Main title */
    h1 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        text-shadow: 0 0 12px rgba(0, 242, 255, 0.6);
    }

    /* Section headers */
    h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 1px;
        color: #00f2ff;
    }

    /* Input fields */
    textarea, input {
        background-color: rgba(31, 41, 51, 0.95) !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #00f2ff55;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #ff0080, #7928ca);
        color: white;
        border-radius: 12px;
        font-weight: bold;
        border: none;
        box-shadow: 0 0 12px #ff008055;
    }

    .stButton > button:hover {
        box-shadow: 0 0 18px #7928ca;
        transform: scale(1.02);
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

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=22)
    education = st.selectbox(
        "Highest Education Level",
        ["Bachelor's", "Master's", "PhD"]
    )

with col2:
    skills = st.text_area(
        "Skills",
        placeholder="e.g. Python, Machine Learning, SQL"
    )
    interests = st.text_area(
        "Interests",
        placeholder="e.g. Data Science, AI"
    )

# -------------------------
# Prediction
# -------------------------
if st.button("🚀 Get Career Recommendation"):
    with st.spinner("Analyzing your profile..."):
        import time
        time.sleep(1)

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

    st.subheader("📊 Confidence Distribution")

    confidence_df = (
        pd.DataFrame({
            "Domain": classes,
            "Confidence": probs * 100
        })
        .sort_values("Confidence", ascending=False)
        .head(5)
    )

    st.bar_chart(confidence_df.set_index("Domain"))


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
# Feedback 
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
