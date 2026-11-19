import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ============================================================
# AI-BASED HEART DISEASE PREDICTION & RISK ANALYSIS SYSTEM
# ============================================================

st.set_page_config(
    page_title="Heart Disease AI Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ AI-Based Heart Disease Prediction & Risk Analysis")
st.write("Machine Learning system for educational heart-disease risk prediction.")

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart_disease_data (1).csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Put 'heart_disease_data (1).csv' in the same folder as app.py.")
    st.stop()

# Clean data
df = df.drop_duplicates().dropna()

if "target" not in df.columns:
    st.error("The CSV must contain a 'target' column.")
    st.stop()

X = df.drop("target", axis=1)
y = df["target"]

# -------------------- Train Models --------------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (LogisticRegression(random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=42), False),
        "Random Forest": (RandomForestClassifier(
            n_estimators=200, max_depth=7, random_state=42
        ), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True),
        "SVM": (SVC(probability=True, random_state=42), True)
    }

    results = []
    trained = {}

    for name, (model, scaled) in models.items():
        if scaled:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

        trained[name] = (model, scaled)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })

    return trained, scaler, X_test, y_test, pd.DataFrame(results)

trained_models, scaler, X_test, y_test, results_df = train_models(X, y)

# Best model based on F1 score
best_model_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]
best_model, best_scaled = trained_models[best_model_name]

if best_scaled:
    test_pred = best_model.predict(scaler.transform(X_test))
    test_prob = best_model.predict_proba(scaler.transform(X_test))[:, 1]
else:
    test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]

# -------------------- Sidebar --------------------
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose Section",
    [
        "🏠 Overview",
        "📊 Data Analysis",
        "🤖 Model Performance",
        "❤️ Patient Prediction"
    ]
)

# -------------------- Overview --------------------
if page == "🏠 Overview":

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Patients", len(df))
    col2.metric("Features", X.shape[1])
    col3.metric("Best Model", best_model_name)
    col4.metric("F1 Score", f"{results_df['F1 Score'].max():.2%}")

    st.markdown("""
    ### Project Pipeline

    **Dataset → Data Cleaning → EDA → Feature Scaling → ML Models → Model Evaluation → Risk Prediction**

    The system compares:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - KNN
    - SVM
    """)

    st.info(
        "⚠️ This application is for educational and research purposes only. "
        "It is not a medical diagnostic tool."
    )

# -------------------- Data Analysis --------------------
elif page == "📊 Data Analysis":

    st.subheader("📊 Exploratory Data Analysis")

    st.write("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Heart Disease Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="target", data=df, ax=ax)
        ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
        ax.set_ylabel("Patients")
        st.pyplot(fig)

    with col2:
        st.write("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Importance")

    rf_model = trained_models["Random Forest"][0]

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

# -------------------- Model Performance --------------------
elif page == "🤖 Model Performance":

    st.subheader("🤖 Machine Learning Model Comparison")

    display_results = results_df.copy()

    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        display_results[col] = display_results[col].round(4)

    st.dataframe(display_results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy Comparison")
    plt.xticks(rotation=25)
    st.pyplot(fig)

    st.success(f"Best model based on F1 Score: {best_model_name}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, test_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -------------------- Patient Prediction --------------------
elif page == "❤️ Patient Prediction":

    st.subheader("❤️ Patient Risk Prediction")

    st.write(
        "Enter patient information below. The model will estimate the "
        "probability of heart disease."
    )

    # Standard UCI/Kaggle heart-disease feature set
    input_values = {}

    col1, col2 = st.columns(2)

    with col1:
        input_values["age"] = st.number_input(
            "Age", min_value=1, max_value=120, value=55
        )

        input_values["sex"] = st.selectbox(
            "Sex", [0, 1],
            format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
        )

        input_values["cp"] = st.selectbox(
            "Chest Pain Type (cp)", [0, 1, 2, 3]
        )

        input_values["trestbps"] = st.number_input(
            "Resting Blood Pressure (trestbps)",
            min_value=50, max_value=250, value=130
        )

        input_values["chol"] = st.number_input(
            "Cholesterol (chol)",
            min_value=50, max_value=700, value=250
        )

        input_values["fbs"] = st.selectbox(
            "Fasting Blood Sugar > 120 (fbs)", [0, 1]
        )

        input_values["restecg"] = st.selectbox(
            "Resting ECG (restecg)", [0, 1, 2]
        )

    with col2:
        input_values["thalach"] = st.number_input(
            "Maximum Heart Rate (thalach)",
            min_value=50, max_value=250, value=150
        )

        input_values["exang"] = st.selectbox(
            "Exercise Induced Angina (exang)", [0, 1]
        )

        input_values["oldpeak"] = st.number_input(
            "ST Depression (oldpeak)",
            min_value=0.0, max_value=10.0, value=1.0, step=0.1
        )

        input_values["slope"] = st.selectbox(
            "Slope", [0, 1, 2]
        )

        input_values["ca"] = st.selectbox(
            "Number of Major Vessels (ca)", [0, 1, 2, 3, 4]
        )

        input_values["thal"] = st.selectbox(
            "Thal", [0, 1, 2, 3]
        )

    if st.button("🔍 Predict Heart Disease", type="primary"):

        # Arrange input exactly like training features
        patient = pd.DataFrame([input_values])

        # Make sure the column order exactly matches the dataset
        patient = patient[X.columns]

        if best_scaled:
            patient_transformed = scaler.transform(patient)
            prediction = best_model.predict(patient_transformed)[0]
            probability = best_model.predict_proba(patient_transformed)[0][1]
        else:
            prediction = best_model.predict(patient)[0]
            probability = best_model.predict_proba(patient)[0][1]

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Heart Disease Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            if probability < 0.30:
                risk = "LOW"
            elif probability < 0.60:
                risk = "MODERATE"
            else:
                risk = "HIGH"

            st.metric("Estimated Risk Level", risk)

        if prediction == 1:
            st.error("⚠️ Model Prediction: Heart Disease Detected")
        else:
            st.success("✅ Model Prediction: No Heart Disease Detected")

        st.info(
            f"Prediction generated using the {best_model_name} model. "
            "This result is an ML estimate and must not be used as a medical diagnosis."
        )

# -------------------- Footer --------------------
st.sidebar.markdown("---")
st.sidebar.caption("AI-Based Heart Disease Prediction System")
st.sidebar.caption("Educational ML Project")
