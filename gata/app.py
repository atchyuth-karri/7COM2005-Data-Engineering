
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Customer Churn Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.subheader("Model Evaluation and Visualisation")

st.write(
    "This dashboard evaluates an existing trained "
    "machine learning model using customer churn data."
)

# ---------------------------------------
# 2. SIDEBAR - FILE UPLOAD
# ---------------------------------------

st.sidebar.header("Upload Files")

model_file = st.sidebar.file_uploader(
    "Upload trained model (.pkl / .joblib)",
    type=["pkl", "joblib"]
)

csv_file = st.sidebar.file_uploader(
    "Upload customer churn dataset (.csv)",
    type=["csv"]
)

# ---------------------------------------
# 3. LOAD MODEL
# ---------------------------------------

def load_model(file):
    try:
        model = joblib.load(file)
        return model
    except Exception:
        file.seek(0)
        return pickle.load(file)


# ---------------------------------------
# 4. LOAD DATASET
# ---------------------------------------

if model_file is not None and csv_file is not None:

    try:
        model = load_model(model_file)
        df = pd.read_csv(csv_file)

        st.success("Model and dataset loaded successfully!")

        # ---------------------------------------
        # 5. DATASET INFORMATION
        # ---------------------------------------

        st.header("1. Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric("Total Features", len(df.columns))

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        st.write("### Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # ---------------------------------------
        # 6. TARGET VARIABLE
        # ---------------------------------------

        st.header("2. Target Variable")

        if "auto_renew" not in df.columns:
            st.error("Target column 'auto_renew' not found.")
            st.stop()

        st.write("Auto-Renew Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.countplot(
            data=df,
            x="auto_renew",
            ax=ax
        )

        ax.set_title("Customer Auto-Renew Distribution")
        ax.set_xlabel("Auto Renew")
        ax.set_ylabel("Number of Customers")

        st.pyplot(fig)
        plt.close(fig)

        # ---------------------------------------
        # 7. PREPARE TEST DATA
        # ---------------------------------------

        st.header("3. Model Evaluation")

        target_col = "auto_renew"

        X = df.drop(
            columns=[target_col, "customer_id"],
            errors="ignore"
        )

        y = df[target_col]

        # Convert categorical variables into numerical
        X_encoded = pd.get_dummies(
            X,
            drop_first=True
        )

        # Align input features with trained model
        expected_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_features is None and hasattr(model, "named_steps"):
            expected_features = getattr(
                model,
                "feature_names_in_",
                None
            )

        if expected_features is not None:

            X_encoded = X_encoded.reindex(
                columns=expected_features,
                fill_value=0
            )

        elif hasattr(model, "n_features_in_"):

            if X_encoded.shape[1] != model.n_features_in_:
                st.error(
                    "Feature count mismatch. "
                    "The uploaded model does not match "
                    "the dataset encoding."
                )
                st.stop()

        else:
            st.error(
                "Cannot verify the model's expected input features. "
                "Please check your saved model."
            )
            st.stop()

        # ---------------------------------------
        # 8. PREDICTIONS
        # ---------------------------------------

        y_pred = model.predict(X_encoded)

        if len(np.unique(y)) != 2:
            st.error("ROC-AUC requires a binary target variable.")
            st.stop()

        # ---------------------------------------
        # 9. EVALUATION METRICS
        # ---------------------------------------

        accuracy = accuracy_score(y, y_pred)

        precision = precision_score(
            y, y_pred, zero_division=0
        )

        recall = recall_score(
            y, y_pred, zero_division=0
        )

        f1 = f1_score(
            y, y_pred, zero_division=0
        )

        y_prob = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_encoded)

            classes = list(model.classes_)

            if 1 in classes:
                positive_index = classes.index(1)
                y_prob = probabilities[:, positive_index]

        elif hasattr(model, "decision_function"):

            y_prob = model.decision_function(X_encoded)

        if y_prob is not None:
            roc_auc = roc_auc_score(y, y_prob)
        else:
            roc_auc = None

        # ---------------------------------------
        # 10. DISPLAY METRICS
        # ---------------------------------------

        st.subheader("Performance Metrics")

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Precision", f"{precision:.4f}")
        c3.metric("Recall", f"{recall:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")

        if roc_auc is not None:
            c5.metric("ROC-AUC", f"{roc_auc:.4f}")
        else:
            c5.metric("ROC-AUC", "N/A")

        # ---------------------------------------
        # 11. CONFUSION MATRIX
        # ---------------------------------------

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Churn (0)", "Renew (1)"],
            yticklabels=["Churn (0)", "Renew (1)"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        # ---------------------------------------
        # 12. CLASSIFICATION REPORT
        # ---------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # ---------------------------------------
        # 13. ROC CURVE
        # ---------------------------------------

        st.header("6. ROC Curve")

        if y_prob is not None:

            fpr, tpr, thresholds = roc_curve(y, y_prob)

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.plot(
                fpr,
                tpr,
                label=f"ROC-AUC = {roc_auc:.4f}"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle="--"
            )

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Receiver Operating Characteristic")

            ax.legend()

            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info("ROC curve unavailable: model has no probability scores.")

        # ---------------------------------------
        # 14. METRICS VISUALISATION
        # ---------------------------------------

        st.header("7. Evaluation Metrics Visualisation")

        metric_names = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ]

        metric_values = [
            accuracy,
            precision,
            recall,
            f1
        ]

        if roc_auc is not None:
            metric_names.append("ROC-AUC")
            metric_values.append(roc_auc)

        fig, ax = plt.subplots(figsize=(10, 5))

        bars = ax.bar(
            metric_names,
            metric_values
        )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance")

        ax.bar_label(
            bars,
            fmt="%.3f"
        )

        st.pyplot(fig)
        plt.close(fig)

        # ---------------------------------------
        # 15. PREDICTION RESULTS
        # ---------------------------------------

        st.header("8. Prediction Results")

        results = pd.DataFrame({
            "Actual": y.reset_index(drop=True),
            "Predicted": pd.Series(y_pred),
            "Correct": (
                y.reset_index(drop=True).to_numpy()
                == y_pred
            )
        })

        if y_prob is not None:
            results["Renew Probability"] = y_prob

        st.dataframe(
            results.head(50),
            use_container_width=True
        )

        csv_output = results.to_csv(index=False)

        st.download_button(
            "Download Evaluation Results",
            data=csv_output,
            file_name="customer_churn_evaluation.csv",
            mime="text/csv"
        )

        st.success("Model evaluation completed successfully!")

    except Exception as e:

        st.error(f"Error: {str(e)}")

else:

    st.info(
        "Please upload both your trained model.pkl "
        "and customer_churn_dataset.csv files."
    )

    st.write("### Required Files")
    st.write("- model.pkl")
    st.write("- customer_churn_dataset.csv")
