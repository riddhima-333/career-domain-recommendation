import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Load dataset
df = pd.read_csv("data/career-data.csv")

# 2️⃣ Career → Domain mapping
career_to_domain = {
    # Data & AI
    "Data Scientist": "Data & AI",
    "Data Analyst": "Data & AI",
    "Data Engineer": "Data & AI",
    "Machine Learning Engineer": "Data & AI",
    "NLP Engineer": "Data & AI",
    "AI Researcher": "Data & AI",
    "Deep Learning Engineer": "Data & AI",
    "Research Scientist": "Data & AI",
    "Biostatistician": "Data & AI",

    # Software Engineering
    "Backend Developer": "Software Engineering",
    "Front-end Developer": "Software Engineering",
    "Full Stack Developer": "Software Engineering",
    "Software Developer": "Software Engineering",
    "Mobile Developer": "Software Engineering",
    "Cloud Engineer": "Software Engineering",
    "DevOps Engineer": "Software Engineering",
    "Embedded Systems Engineer": "Software Engineering",

    # Design
    "UX Designer": "Design",
    "UX Researcher": "Design",
    "Graphic Designer": "Design",

    # Business & Management
    "Business Analyst": "Business & Management",
    "Project Manager": "Business & Management",
    "Financial Analyst": "Business & Management",

    # Other (merged weak classes)
    "Digital Marketer": "Other",
    "Content Strategist": "Other",
    "Cybersecurity Analyst": "Other",
    "Cybersecurity Specialist": "Other",
    "Research Analyst": "Other"
}


# 3️⃣ Create NEW target column
df["Career_Domain"] = df["Recommended_Career"].map(career_to_domain)

# Optional safety check
df = df.dropna(subset=["Career_Domain"])

# 4️⃣ Define features & TARGET
X = df[["Age", "Education", "Skills", "Interests"]]
y = df["Career_Domain"]   # ✅ DOMAIN-LEVEL TARGET

# 5️⃣ Column groups
numeric_features = ["Age"]
categorical_features = ["Education"]

# 6️⃣ Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["Age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Education"]),
        ("skills_tfidf", TfidfVectorizer(
            max_features=300,
            stop_words="english",
            ngram_range=(1, 2)
        ), "Skills"),
        ("interests_tfidf", TfidfVectorizer(
            max_features=300,
            stop_words="english",
            ngram_range=(1, 2)
        ), "Interests"),
    ]
)

# 7️⃣ Model
classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# 8️⃣ Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

# 9️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔟 Train
model.fit(X_train, y_train)

import joblib

joblib.dump(model, "models/career_domain_model.pkl")
print("Model saved successfully")

# 1️⃣1️⃣ Predict & evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
