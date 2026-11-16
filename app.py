# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")
st.markdown(
    "Predict customer auto-renewal using a trained "
    "Gradient Boosting Machine Learning model."
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "customer_churn_dataset(1).csv"
MODEL_PATH = BASE_DIR / "gradient_boosting_model (1).pkl"

# ==========================================================
# LOAD DATASET AND MODEL
# ==========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_dataset()
    model = load_model()

except Exception as e:
    st.error(f"Error loading dataset or model: {e}")
    st.stop()


if "auto_renew" not in df.columns:
    st.error("The dataset must contain an auto_renew target column.")
    st.stop()


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def prepare_features(data, expected_features):

    X = pd.DataFrame(index=data.index)

    # Original dataset to model feature mapping
    mapping = {
        "duration_days": "duration_days",
        "total_orders": "total_orders",
        "total_spend_pence": "total_spend",
        "avg_order_value_pence": "avg_basket_value",
        "total_items_purchased": "total_items",
        "unique_products_purchased": "unique_products",
        "completed_deliveries": "completed_deliveries",
        "unfulfilled_orders": "unfulfilled_deliveries",
        "delayed_deliveries": "delayed_deliveries",
        "delivery_delay_ratio": "delay_ratio",
        "total_crm_interactions": "total_contacts",
        "crm_complaints_count": "complaints",
        "crm_queries_count": "queries",
        "crm_email_count": "email_contacts",
        "crm_phone_count": "phone_contacts",
        "crm_livechat_count": "chat_contacts",
        "crm_complaint_ratio": "complaint_ratio",
        "total_reviews": "total_reviews",
        "avg_star_rating": "avg_rating",
        "high_star_reviews_count": "positive_reviews",
        "has_reviews": "has_reviewed",
        "total_web_sessions": "total_web_visits",
        "days_since_last_session": "recency_days",
        "session_frequency_per_month": "session_frequency",
        "order_frequency_per_month": "order_frequency",
        "nps_recommend": "nps_recommend",
        "satisfaction_score": "satisfaction_score"
    }

    for model_feature, dataset_feature in mapping.items():

        if dataset_feature in data.columns:

            X[model_feature] = data[dataset_feature]

            if model_feature in [
                "total_spend_pence",
                "avg_order_value_pence"
            ]:
                X[model_feature] = X[model_feature] * 100

    # Derived features

    if "total_items" in data.columns and "total_orders" in data.columns:

        X["avg_items_per_order"] = (
            data["total_items"] /
            data["total_orders"].replace(0, np.nan)
        )

    if "total_delay_hours" in data.columns and "delayed_deliveries" in data.columns:

        X["avg_delivery_delay_hours"] = (
            data["total_delay_hours"] /
            data["delayed_deliveries"].replace(0, np.nan)
        )

    if "total_reviews" in data.columns and "positive_reviews" in data.columns:

        X["low_star_reviews_count"] = (
            data["total_reviews"] - data["positive_reviews"]
        )

    # Categorical encoding using the model's expected columns

    categorical_columns = {
        "subscription_tier": "subscription_tier_",
        "delivery_service": "delivery_service_",
        "income_bracket": "reported_income_"
    }

    for original_column, prefix in categorical_columns.items():

        if original_column in data.columns:

            for feature in expected_features:

                if feature.startswith(prefix):

                    category = feature.replace(prefix, "")

                    X[feature] = (
                        data[original_column].astype(str) == category
                    ).astype(int)

    # Some training features are not available in this CSV.
    # Fill unavailable features with zero rather than inventing values.

    for feature in expected_features:

        if feature not in X.columns:
            X[feature] = 0

    X = X[expected_features]

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X


# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.sidebar.title("⚙️ Model Information")

st.sidebar.success("Gradient Boosting Model Loaded")

st.sidebar.write("**Algorithm:** Gradient Boosting Classifier")

st.sidebar.write(f"**Dataset Records:** {len(df)}")

st.sidebar.write(f"**Dataset Features:** {len(df.columns) - 2}")

st.sidebar.write(f"**Model Features:** {model.n_features_in_}")

st.sidebar.write("**Target:** auto_renew")

st.sidebar.write("**Classes:** 0 = No, 1 = Yes")


# ==========================================================
# PREPARE DATA FOR EVALUATION
# ==========================================================

expected_features = list(model.feature_names_in_)

X = prepare_features(df, expected_features)

y = df["auto_renew"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Evaluate the existing model. Do not train a new model.

try:

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test)[:, 1]
    else:
        y_probability = None

except Exception as e:

    st.error(f"Model evaluation error: {e}")
    st.stop()


# ==========================================================
# NAVIGATION
# ==========================================================

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "🏠 Home",
        "🔮 Customer Prediction",
        "📈 Model Evaluation",
        "📊 Visualizations",
        "📁 Dataset Explorer"
    ]
)


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.header("Customer Churn Prediction Dashboard")

    st.write(
        "This application uses an existing trained Gradient Boosting "
        "model to predict whether a customer will renew their subscription."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Renewed Customers",
        int((y == 1).sum())
    )

    col3.metric(
        "Not Renewed",
        int((y == 0).sum())
    )

    col4.metric(
        "Test Records",
        len(y_test)
    )

    st.divider()

    st.subheader("Project Features")

    st.markdown("""
    - Customer information input.
    - Individual customer churn prediction.
    - Renewal probability.
    - Gradient Boosting model evaluation.
    - Accuracy, Precision, Recall and F1-score.
    - Confusion matrix.
    - Classification report.
    - ROC curve and AUC score.
    - Feature importance visualization.
    - Customer dataset exploration.
    - Download prediction results.
    """)

    st.info(
        "Select Customer Prediction from the sidebar to enter customer "
        "details and receive a prediction."
    )

    st.warning(
        "The uploaded dataset and model have different feature schemas. "
        "Some model features are unavailable or have different category "
        "values in the dataset. Missing features are filled with zero, "
        "so evaluation results should be treated as provisional until "
        "the original training preprocessing is confirmed."
    )


# ==========================================================
# CUSTOMER PREDICTION PAGE
# ==========================================================

elif page == "🔮 Customer Prediction":

    st.header("🔮 Predict Customer Auto-Renewal")

    st.write(
        "Enter customer details below to predict subscription renewal."
    )

    # Default values from the first customer record

    default_row = df.iloc[0]

    input_data = {}

    with st.form("prediction_form"):

        st.subheader("Customer Information")

        columns = st.columns(2)

        input_columns = [
            c for c in df.columns
            if c not in ["customer_id", "auto_renew"]
        ]

        for i, column in enumerate(input_columns):

            with columns[i % 2]:

                if pd.api.types.is_numeric_dtype(df[column]):

                    minimum = float(df[column].min())
                    maximum = float(df[column].max())
                    default = float(default_row[column])

                    if pd.api.types.is_integer_dtype(df[column]):

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default),
                            step=1
                        )

                    else:

                        input_data[column] = st.number_input(
                            column.replace("_", " ").title(),
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            format="%.4f"
                        )

                else:

                    options = sorted(
                        df[column].dropna().astype(str).unique().tolist()
                    )

                    default_value = str(default_row[column])

                    input_data[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        options,
                        index=(
                            options.index(default_value)
                            if default_value in options else 0
                        )
                    )

        submitted = st.form_submit_button(
            "🔮 Predict Customer Renewal",
            use_container_width=True
        )

    if submitted:

        customer_df = pd.DataFrame([input_data])

        customer_features = prepare_features(
            customer_df,
            expected_features
        )

        try:

            prediction = model.predict(customer_features)[0]

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    customer_features
                )[0]

                classes = list(model.classes_)

                probability_yes = probabilities[classes.index(1)]
                probability_no = probabilities[classes.index(0)]

            else:

                probability_yes = None
                probability_no = None

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

                st.success("✅ Customer is predicted to RENEW.")

            else:

                st.error("❌ Customer is predicted NOT TO RENEW.")

            if probability_yes is not None:

                col1, col2 = st.columns(2)

                col1.metric(
                    "Renewal Probability",
                    f"{probability_yes * 100:.2f}%"
                )

                col2.metric(
                    "Non-Renewal Probability",
                    f"{probability_no * 100:.2f}%"
                )

                st.progress(float(probability_yes))

                st.caption("Renewal probability")

            # Display customer prediction

            result = customer_df.copy()

            result["Predicted_auto_renew"] = int(prediction)

            if probability_yes is not None:

                result["Renewal_probability"] = probability_yes

                result["Non_renewal_probability"] = probability_no

            st.subheader("Customer Prediction Details")

            st.dataframe(
                result,
                use_container_width=True
            )

            # Download prediction

            csv_result = result.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction Result",
                data=csv_result,
                file_name="customer_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ==========================================================
# MODEL EVALUATION PAGE
# ==========================================================

elif page == "📈 Model Evaluation":

    st.header("📈 Gradient Boosting Model Evaluation")

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    # ROC AUC

    if y_probability is not None:

        auc_score = roc_auc_score(
            y_test,
            y_probability
        )

        st.metric(
            "ROC-AUC Score",
            f"{auc_score:.4f}"
        )

    # Confusion matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Renewed", "Renewed"],
        yticklabels=["Not Renewed", "Renewed"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # Classification report

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Renewed", "Renewed"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True
    )

    # Evaluation results table

    st.subheader("Performance Summary")

    evaluation = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    st.dataframe(
        evaluation,
        use_container_width=True
    )


# ==========================================================
# VISUALIZATION PAGE
# ==========================================================

elif page == "📊 Visualizations":

    st.header("📊 Model Performance Visualizations")

    # 1. Target distribution

    st.subheader("1. Customer Renewal Distribution")

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.countplot(
        x=y,
        hue=y,
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Renewed", "Renewed"])
    ax.set_xlabel("Customer Status")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Renewal Distribution")

    st.pyplot(fig)

    st.divider()

    # 2. Confusion Matrix

    st.subheader("2. Confusion Matrix Heatmap")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.divider()

    # 3. ROC Curve

    st.subheader("3. ROC Curve")

    if y_probability is not None:

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_probability
        )

        auc_score = roc_auc_score(
            y_test,
            y_probability
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            fpr,
            tpr,
            label=f"Gradient Boosting (AUC = {auc_score:.3f})"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend()

        st.pyplot(fig)

    else:

        st.info("This model does not support probability predictions.")

    st.divider()

    # 4. Evaluation Metrics

    st.subheader("4. Performance Metrics Comparison")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, zero_division=0),
            recall_score(y_test, y_pred, zero_division=0),
            f1_score(y_test, y_pred, zero_division=0)
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=metrics,
        x="Metric",
        y="Score",
        hue="Metric",
        palette="viridis",
        legend=False,
        ax=ax
    )

    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Metrics")

    st.pyplot(fig)

    st.divider()

    # 5. Feature Importance

    st.subheader("5. Feature Importance")

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": expected_features,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        ).head(15)

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.barplot(
            data=importance,
            x="Importance",
            y="Feature",
            hue="Feature",
            palette="magma",
            legend=False,
            ax=ax
        )

        ax.set_title("Top 15 Important Features")

        st.pyplot(fig)

        st.dataframe(
            importance,
            use_container_width=True
        )

    else:

        st.info("Feature importance is not available for this model.")


# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page == "📁 Dataset Explorer":

    st.header("📁 Customer Churn Dataset Explorer")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])

    col2.metric("Total Columns", df.shape[1])

    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(13, 8))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=False,
        ax=ax
    )

    ax.set_title("Feature Correlation Matrix")

    st.pyplot(fig)

    st.subheader("Download Dataset")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Dataset",
        data=csv_data,
        file_name="customer_churn_dataset.csv",
        mime="text/csv"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Machine Learning | Gradient Boosting Classifier"
)