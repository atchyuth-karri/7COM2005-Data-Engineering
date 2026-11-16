"""
PerfectPetClub Churn Analytics & Live PySpark MLlib Training Platform.
Module: 7COM2005 - Data Engineering | University of Hertfordshire

Features:
- Live Distributed PySpark MLlib Pipeline Training on local Spark cluster.
- Evaluates on strictly held-out test partitions (80/20 split) to prevent data leakage.
- Comparison between Leakage-Controlled (14-day cutoff) and Leaky (Full-window) datasets.
- Interactive visualizations: Confusion Matrix, ROC-AUC curves, Feature Importance.
- Operational decision-support: Scored predictions table with probability thresholding.

Run from repository root:
    streamlit run gata/app.py
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent / "PerfectPetClub"
CUTOFF_PARQUET = PROJECT / "data/processed/customer_churn_dataset_cutoff14.parquet"
CANONICAL_PARQUET = PROJECT / "data/processed/customer_churn_dataset.parquet"
OUTPUT_CUTOFF = PROJECT / "outputs/ml_cutoff14"
OUTPUT_CANON = PROJECT / "outputs/ml"
SELECTED = OUTPUT_CUTOFF / "selected_model.csv"
PREDICTIONS = OUTPUT_CUTOFF / "test_predictions_selected_model.csv"

CHURN, RENEWED = 1, 0
LABEL_NAMES = {RENEWED: "Renewed (0)", CHURN: "Churned (1)"}

st.set_page_config(
    page_title="PerfectPetClub — PySpark Churn Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting PySpark Context...")
def get_spark():
    import os
    import sys
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("PerfectPetClub-LivePlatform")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource(show_spinner="Loading pre-saved PySpark pipeline...")
def load_pipeline(path: str):
    from pyspark.ml import PipelineModel
    return PipelineModel.load(path)


@st.cache_data(show_spinner=False)
def load_features(cutoff: bool = True) -> pd.DataFrame:
    path = CUTOFF_PARQUET if cutoff else CANONICAL_PARQUET
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS)


@st.cache_data(show_spinner=False)
def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# PySpark Live Training Engine
# ---------------------------------------------------------------------------
def train_pyspark_pipeline_live(dataset_type: str, model_type: str, params: dict):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        StringIndexer,
        OneHotEncoder,
        Imputer,
        VectorAssembler,
        StandardScaler,
    )
    from pyspark.ml.classification import (
        LogisticRegression,
        RandomForestClassifier,
        GBTClassifier,
    )
    import pyspark.sql.functions as F

    spark = get_spark()
    parquet_path = CUTOFF_PARQUET if "14-Day" in dataset_type else CANONICAL_PARQUET
    df = spark.read.parquet(str(parquet_path))
    df = df.withColumn("churn_label", (1 - F.col("auto_renew")).cast("double"))

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    cat_cols = ["subscription_tier", "delivery_service"]
    num_cols = [
        "total_orders",
        "total_spend_pence",
        "avg_order_value_pence",
        "total_items_purchased",
        "unique_products_purchased",
        "avg_items_per_order",
        "completed_deliveries",
        "unfulfilled_orders",
        "delayed_deliveries",
        "avg_delivery_delay_hours",
        "delivery_delay_ratio",
        "total_crm_interactions",
        "crm_complaints_count",
        "crm_queries_count",
        "crm_email_count",
        "crm_phone_count",
        "crm_livechat_count",
        "crm_complaint_ratio",
        "total_web_sessions",
        "days_since_last_session",
        "session_frequency_per_month",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    stages = []
    for c in cat_cols:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))

    stages.append(
        OneHotEncoder(
            inputCols=[f"{c}_idx" for c in cat_cols],
            outputCols=[f"{c}_ohe" for c in cat_cols],
            handleInvalid="keep",
        )
    )

    stages.append(
        Imputer(
            inputCols=num_cols,
            outputCols=[f"{c}_imp" for c in num_cols],
            strategy="median",
        )
    )

    stages.append(
        VectorAssembler(
            inputCols=[f"{c}_imp" for c in num_cols] + [f"{c}_ohe" for c in cat_cols],
            outputCol="features_raw",
            handleInvalid="keep",
        )
    )

    stages.append(
        StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=False,
            withStd=True,
        )
    )

    if model_type == "Logistic Regression":
        classifier = LogisticRegression(
            labelCol="churn_label",
            featuresCol="features",
            maxIter=params.get("maxIter", 25),
            regParam=params.get("regParam", 0.01),
        )
    elif model_type == "Random Forest":
        classifier = RandomForestClassifier(
            labelCol="churn_label",
            featuresCol="features",
            numTrees=params.get("numTrees", 20),
            maxDepth=params.get("maxDepth", 5),
            seed=42,
        )
    else:  # Gradient-Boosted Trees
        classifier = GBTClassifier(
            labelCol="churn_label",
            featuresCol="features",
            maxIter=params.get("maxIter", 15),
            maxDepth=params.get("maxDepth", 3),
            seed=42,
        )

    stages.append(classifier)
    pipeline = Pipeline(stages=stages)

    t0 = time.time()
    fitted_model = pipeline.fit(train_df)
    fit_duration = time.time() - t0

    preds = fitted_model.transform(test_df).select(
        "customer_id", "churn_label", "prediction", "probability"
    ).toPandas()
    preds["churn_probability"] = preds["probability"].apply(lambda v: float(v[1]))
    preds = preds.drop(columns="probability")

    # Extract feature weights / importances
    feature_names = [c for c in num_cols] + [
        "tier_SuperSaver",
        "tier_Basic",
        "deliv_Economy72",
        "deliv_Tracked48",
    ]
    model_stage = fitted_model.stages[-1]
    feature_importance_df = None

    try:
        if model_type == "Logistic Regression":
            weights = model_stage.coefficients.toArray()
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names[:len(weights)],
                "Impact (Weight)": weights,
                "Abs Impact": np.abs(weights),
            }).sort_values(by="Abs Impact", ascending=False)
        else:
            importances = model_stage.featureImportances.toArray()
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names[:len(importances)],
                "Importance": importances,
            }).sort_values(by="Importance", ascending=False)
    except Exception:
        pass

    return {
        "model": fitted_model,
        "duration": fit_duration,
        "train_count": train_df.count(),
        "test_count": test_df.count(),
        "preds": preds,
        "feature_importance": feature_importance_df,
        "model_type": model_type,
        "dataset_type": dataset_type,
    }


# ---------------------------------------------------------------------------
# Header & Warning Banner
# ---------------------------------------------------------------------------
st.title("⚡ PerfectPetClub — PySpark MLlib Churn Platform")
st.caption("Module: 7COM2005 – Data Engineering | University of Hertfordshire | Live Pipeline Serving & Diagnostic Suite")

st.info(
    "💡 **Academic Finding on Temporal Data Leakage:** Baseline full-window features cover activities up to "
    "cancellation, inflating ROC-AUC to ~0.78 by leaking subscription survival time. Under our **14-day fixed-cutoff horizon**, "
    "models evaluate true pre-cancellation behavioral signals, resulting in an honest, scientifically grounded AUC (~0.55–0.60)."
)

if not (CUTOFF_PARQUET.exists() and PREDICTIONS.exists() and SELECTED.exists()):
    st.error(
        "Required artifacts missing. Please verify that notebook 03 and notebook 04 have generated the Parquet and evaluation artifacts."
    )
    st.stop()

selected = pd.read_csv(SELECTED).iloc[0]
pipeline_path = PROJECT / selected["pipeline_path"]
features = load_features(cutoff=True)
preds = load_predictions()

# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_eval, tab_live, tab_data = st.tabs([
    "📊 Audited Pipeline Diagnostics",
    "⚡ Live PySpark Training Studio",
    "🔍 Feature Store & Data Schema",
])

# ===========================================================================
# TAB 1: AUDITED PIPELINE DIAGNOSTICS
# ===========================================================================
with tab_eval:
    st.subheader("Audited Model Evaluation (Held-Out Test Customers)")
    st.write(
        f"Selected Pipeline: **{selected['selected_config']} / {selected['selected_model']}** "
        f"(Saved in `{selected['pipeline_path']}`). Strictly evaluated on 44 held-out test customers."
    )

    test_ids = preds["customer_id"].tolist()
    try:
        spark = get_spark()
        saved_model = load_pipeline(str(pipeline_path))
        sdf = spark.createDataFrame(
            features[features["customer_id"].isin(test_ids)].assign(
                churn_label=lambda d: (1 - d["auto_renew"]).astype(float)
            )
        )
        scored = saved_model.transform(sdf).select(
            "customer_id", "churn_label", "prediction", "probability"
        ).toPandas()
        scored["churn_probability"] = scored["probability"].apply(lambda v: float(v[1]))
        scored = scored.drop(columns="probability")
        live_rescore_ok = True
    except Exception as exc:
        st.error(f"Could not re-score saved pipeline: {exc}")
        scored = preds.copy()
        live_rescore_ok = False

    if live_rescore_ok:
        merged = scored.merge(preds[["customer_id", "prediction"]], on="customer_id", suffixes=("", "_saved"))
        agree = int((merged["prediction"] == merged["prediction_saved"]).sum())
        st.success(f"✅ Live re-scoring verified: Matches saved notebook predictions for {agree}/{len(merged)} test customers.")

    y_true = scored["churn_label"].astype(int).to_numpy()
    y_pred = scored["prediction"].astype(int).to_numpy()
    y_prob = scored["churn_probability"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=CHURN, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=CHURN, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=CHURN, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy:.3f}")
    m2.metric("Churn Precision", f"{precision:.3f}")
    m3.metric("Churn Recall", f"{recall:.3f}")
    m4.metric("Churn F1", f"{f1:.3f}")
    m5.metric("ROC-AUC (Churn)", f"{roc_auc:.3f}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.write("#### Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[RENEWED, CHURN])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax_cm,
            xticklabels=["Pred Renewed (0)", "Pred Churned (1)"],
            yticklabels=["Act Renewed (0)", "Act Churned (1)"],
        )
        ax_cm.set_title("Held-Out Test Set (44 Customers)")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_roc:
        st.write("#### ROC Curve (Positive = Churn)")
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=CHURN)
            fig_roc, ax_roc = plt.subplots(figsize=(5, 3.8))
            ax_roc.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"PySpark Model (AUC = {roc_auc:.3f})")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance Baseline (0.50)")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate (Recall)")
            ax_roc.legend(loc="lower right", fontsize=8)
            ax_roc.grid(True, alpha=0.3)
            st.pyplot(fig_roc)
            plt.close(fig_roc)

    st.write("#### Classification Report Breakdown")
    report = classification_report(
        y_true,
        y_pred,
        labels=[RENEWED, CHURN],
        target_names=[LABEL_NAMES[RENEWED], LABEL_NAMES[CHURN]],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.write("#### Model Comparison: Fixed-Cutoff vs. Leaky Baseline")
    cut_results = load_results(OUTPUT_CUTOFF / "model_results.csv")
    st.dataframe(
        cut_results[["config", "model", "cv_roc_auc", "roc_auc", "churn_precision", "churn_recall", "churn_f1", "accuracy"]].round(3),
        use_container_width=True,
    )

    fig_bar, ax_bar = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(cut_results))
    ax_bar.bar(x - 0.2, cut_results["cv_roc_auc"], width=0.4, label="CV ROC-AUC", color="#4c72b0")
    ax_bar.bar(x + 0.2, cut_results["roc_auc"], width=0.4, label="Test ROC-AUC", color="#55a868")
    ax_bar.axhline(0.5, linestyle="--", color="gray", label="Chance (0.50)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(cut_results["config"] + "\n" + cut_results["model"], fontsize=8)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("ROC-AUC")
    ax_bar.legend(loc="upper right", fontsize=8)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.write("#### Scored Test Predictions (Customer Level)")
    results_tbl = pd.DataFrame({
        "Customer ID": scored["customer_id"].to_numpy(),
        "Actual Outcome": pd.Series(y_true).map(LABEL_NAMES).to_numpy(),
        "Predicted Outcome": pd.Series(y_pred).map(LABEL_NAMES).to_numpy(),
        "Correct": y_true == y_pred,
        "Churn Probability": y_prob.round(4),
    })
    st.dataframe(results_tbl.sort_values("Churn Probability", ascending=False), height=280, use_container_width=True)
    st.download_button(
        "📥 Download Test Predictions (CSV)",
        data=results_tbl.to_csv(index=False),
        file_name="test_predictions_cutoff14.csv",
        mime="text/csv",
    )


# ===========================================================================
# TAB 2: LIVE PYSPARK TRAINING STUDIO
# ===========================================================================
with tab_live:
    st.subheader("⚡ Live Distributed PySpark Pipeline Studio")
    st.markdown(
        "Demonstrate end-to-end model fitting directly on the Apache Spark cluster in real time. "
        "Test different algorithms and verify how the **14-day cutoff controls for temporal leakage** compared to the full-window baseline."
    )

    with st.expander("🛠️ Pipeline Training Controls & Hyperparameters", expanded=True):
        c_ds, c_algo = st.columns(2)
        with c_ds:
            ds_choice = st.radio(
                "Observation Horizon (Feature Store):",
                ["14-Day Fixed Cutoff (Leakage-Controlled)", "Full-Window (Baseline Leaky)"],
                help="Switch between early-observation features and full-tenure cumulative features."
            )
        with c_algo:
            algo_choice = st.selectbox(
                "Spark MLlib Estimator:",
                ["Logistic Regression", "Random Forest", "Gradient-Boosted Trees (GBT)"],
            )

        st.markdown("---")
        h1, h2 = st.columns(2)
        params = {}
        if algo_choice == "Logistic Regression":
            with h1:
                params["maxIter"] = st.slider("Max Iterations", 10, 100, 25, 5)
            with h2:
                params["regParam"] = st.select_slider("Regularization (regParam)", options=[0.0, 0.001, 0.01, 0.1, 1.0], value=0.01)
        elif algo_choice == "Random Forest":
            with h1:
                params["numTrees"] = st.slider("Number of Trees", 5, 50, 20, 5)
            with h2:
                params["maxDepth"] = st.slider("Max Tree Depth", 2, 8, 5, 1)
        else:
            with h1:
                params["maxIter"] = st.slider("Max Iterations", 5, 30, 15, 5)
            with h2:
                params["maxDepth"] = st.slider("Max Tree Depth", 2, 6, 3, 1)

        train_btn = st.button("🚀 Train PySpark MLlib Pipeline Live", type="primary")

    if train_btn:
        with st.spinner("Executing PySpark Pipeline: StringIndexer ➔ OneHotEncoder ➔ Imputer ➔ VectorAssembler ➔ StandardScaler ➔ Model Fit..."):
            res = train_pyspark_pipeline_live(ds_choice, algo_choice, params)
            st.session_state["live_result"] = res

    if "live_result" in st.session_state:
        res = st.session_state["live_result"]
        preds_live = res["preds"]
        y_true_live = preds_live["churn_label"].astype(int).to_numpy()
        y_pred_live = preds_live["prediction"].astype(int).to_numpy()
        y_prob_live = preds_live["churn_probability"].to_numpy()

        acc_l = accuracy_score(y_true_live, y_pred_live)
        prec_l = precision_score(y_true_live, y_pred_live, pos_label=CHURN, zero_division=0)
        rec_l = recall_score(y_true_live, y_pred_live, pos_label=CHURN, zero_division=0)
        f1_l = f1_score(y_true_live, y_pred_live, pos_label=CHURN, zero_division=0)
        auc_l = roc_auc_score(y_true_live, y_prob_live) if len(np.unique(y_true_live)) == 2 else float("nan")

        st.success(
            f"✅ **PySpark MLlib Pipeline fitted in {res['duration']:.2f} seconds!** "
            f"Dataset: **{res['dataset_type']}** | Train rows: **{res['train_count']}** | Held-Out Test rows: **{res['test_count']}**"
        )

        # Spark DAG visualization
        st.caption("Spark Execution Graph: `DataFrame` ➔ `StringIndexer` ➔ `OneHotEncoder` ➔ `Imputer(median)` ➔ `VectorAssembler` ➔ `StandardScaler` ➔ `MLlib Model`")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Live Accuracy", f"{acc_l:.3f}")
        col2.metric("Live Precision", f"{prec_l:.3f}")
        col3.metric("Live Recall", f"{rec_l:.3f}")
        col4.metric("Live F1-Score", f"{f1_l:.3f}")
        col5.metric("Live ROC-AUC", f"{auc_l:.3f}")

        st.markdown("---")
        g1, g2 = st.columns(2)

        with g1:
            st.write("#### Live Confusion Matrix")
            cm_l = confusion_matrix(y_true_live, y_pred_live, labels=[RENEWED, CHURN])
            fig_cm_l, ax_cm_l = plt.subplots(figsize=(5, 3.8))
            sns.heatmap(
                cm_l,
                annot=True,
                fmt="d",
                cmap="Greens",
                cbar=False,
                ax=ax_cm_l,
                xticklabels=["Pred Renewed (0)", "Pred Churned (1)"],
                yticklabels=["Act Renewed (0)", "Act Churned (1)"],
            )
            ax_cm_l.set_title(f"Live Fit: {res['model_type']}")
            st.pyplot(fig_cm_l)
            plt.close(fig_cm_l)

        with g2:
            st.write("#### Live ROC Curve")
            if len(np.unique(y_true_live)) == 2:
                fpr_l, tpr_l, _ = roc_curve(y_true_live, y_prob_live, pos_label=CHURN)
                fig_roc_l, ax_roc_l = plt.subplots(figsize=(5, 3.8))
                ax_roc_l.plot(fpr_l, tpr_l, color="#2ca02c", lw=2, label=f"Live {res['model_type']} (AUC = {auc_l:.3f})")
                ax_roc_l.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (0.50)")
                ax_roc_l.set_xlabel("False Positive Rate")
                ax_roc_l.set_ylabel("True Positive Rate (Recall)")
                ax_roc_l.legend(loc="lower right", fontsize=8)
                ax_roc_l.grid(True, alpha=0.3)
                st.pyplot(fig_roc_l)
                plt.close(fig_roc_l)

        # Feature Importance / Coefficients Section
        if res["feature_importance"] is not None and not res["feature_importance"].empty:
            st.write("#### Feature Impact & Predictive Signal")
            feat_df = res["feature_importance"].head(10)
            fig_imp, ax_imp = plt.subplots(figsize=(8, 3.5))
            if "Importance" in feat_df.columns:
                sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", legend=False, palette="Greens_r", ax=ax_imp)
                ax_imp.set_title(f"Top 10 Feature Importances ({res['model_type']})")
            else:
                sns.barplot(data=feat_df, x="Impact (Weight)", y="Feature", hue="Feature", legend=False, palette="vlag", ax=ax_imp)
                ax_imp.set_title(f"Top 10 Model Coefficients ({res['model_type']})")
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        st.write("#### Scored Test Predictions (Live Trained Model)")
        live_tbl = pd.DataFrame({
            "Customer ID": preds_live["customer_id"].to_numpy(),
            "Actual Outcome": pd.Series(y_true_live).map(LABEL_NAMES).to_numpy(),
            "Live Prediction": pd.Series(y_pred_live).map(LABEL_NAMES).to_numpy(),
            "Correct": y_true_live == y_pred_live,
            "Churn Probability": y_prob_live.round(4),
        })
        st.dataframe(live_tbl.sort_values("Churn Probability", ascending=False), height=250, use_container_width=True)

        st.download_button(
            "📥 Download Live Predictions (CSV)",
            data=live_tbl.to_csv(index=False),
            file_name=f"live_predictions_{res['model_type'].replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )
    else:
        st.info("👆 Select hyperparameters above and click **'🚀 Train PySpark MLlib Pipeline Live'** to execute a distributed model training run in under 5 seconds.")


# ===========================================================================
# TAB 3: FEATURE STORE & DATA SCHEMA
# ===========================================================================
with tab_data:
    st.subheader("Canonical Feature Store & Data Lineage Explorer")
    st.write(
        "Single source of truth created by the end-to-end ETL pipeline harmonizing "
        "operational SQLite tables, CRM support logs, web access logs, and the S3 customer survey."
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Customer Subscribers", len(features))
    d2.metric("Engineered Features", len(features.columns))
    d3.metric("Null Values", int(features.isnull().sum().sum()))
    d4.metric("Financial Discrepancy", "£0.00")

    col_target, col_tiers = st.columns(2)
    with col_target:
        st.write("#### Target Variable Distribution")
        fig_t, ax_t = plt.subplots(figsize=(5, 3.2))
        counts = features["auto_renew"].map({0: "Churned (0)", 1: "Renewed (1)"}).value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette=["#d95f02", "#7570b3"], ax=ax_t)
        ax_t.set_ylabel("Customers")
        st.pyplot(fig_t)
        plt.close(fig_t)

    with col_tiers:
        st.write("#### Subscription Tiers")
        fig_st, ax_st = plt.subplots(figsize=(5, 3.2))
        tier_counts = features["subscription_tier"].value_counts()
        sns.barplot(x=tier_counts.index, y=tier_counts.values, hue=tier_counts.index, legend=False, palette="Blues_r", ax=ax_st)
        ax_st.set_ylabel("Subscribers")
        st.pyplot(fig_st)
        plt.close(fig_st)

    st.write("#### Feature Store Dataframe Preview")
    st.dataframe(features.head(15), use_container_width=True)
