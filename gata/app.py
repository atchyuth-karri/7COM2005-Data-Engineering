"""
PerfectPetClub churn dashboard (corrected).

What changed from the original team app:
- No manual uploads. The app reads the canonical project files directly.
- The model is the saved PySpark pipeline from notebook 04 (fixed-cutoff features),
  not the scikit-learn pickle trained on the mock CSV.
- Metrics are computed on the held-out test customers only, not the whole dataset.
- Churn (auto_renew = 0) is the positive class for precision, recall, F1 and ROC.
- Confusion-matrix labels match the encoding: 0 = renewed, 1 = churned.
- A leakage warning is shown, because the original full-window features leak outcome information.

Run from the repository root:
    streamlit run gata/app.py
Requires Java 17 or 21 (for PySpark) and the packages in gata/requirements.txt.
"""

from pathlib import Path

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

st.set_page_config(page_title="PerfectPetClub Churn Dashboard", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Starting Spark...")
def get_spark():
    import os
    import sys

    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (SparkSession.builder.master("local[*]").appName("PerfectPetClub-Dashboard")
             .config("spark.driver.memory", "2g")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource(show_spinner="Loading saved PySpark pipeline...")
def load_pipeline(path: str):
    from pyspark.ml import PipelineModel

    return PipelineModel.load(path)


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(CUTOFF_PARQUET)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS)


@st.cache_data(show_spinner=False)
def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Header and warning
# ---------------------------------------------------------------------------

st.title("📊 Customer Churn Prediction Dashboard")
st.subheader("Model evaluation on held-out customers")

st.warning(
    "**Read before using these numbers.** The original features covered each subscription's whole "
    "window, which ends at cancellation for churned customers, so they leaked outcome information. "
    "This dashboard uses the leakage-controlled features (events before start + 14 days). With those "
    "features the models are close to chance, so **this is not a usable renewal predictor**. "
    "Churn is `auto_renew = 0`, a proxy that has not been checked against cancellation records."
)

if not (CUTOFF_PARQUET.exists() and PREDICTIONS.exists() and SELECTED.exists()):
    st.error(
        "Required outputs are missing. Run `PerfectPetClub/notebooks/03_Fixed_Cutoff_Features.ipynb` "
        "and then `04_Cutoff_Machine_Learning.ipynb`."
    )
    st.stop()

selected = pd.read_csv(SELECTED).iloc[0]
pipeline_path = PROJECT / selected["pipeline_path"]
features = load_features()
preds = load_predictions()

# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------

st.header("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(features))
c2.metric("Columns", len(features.columns))
c3.metric("Missing values", int(features.isnull().sum().sum()))
st.caption("Undefined rates (e.g. average order value with no orders) would be imputed inside the pipeline from training data only. This cutoff dataset has no such gaps.")

with st.expander("Dataset preview"):
    st.dataframe(features.head(10))

# ---------------------------------------------------------------------------
# 2. Target
# ---------------------------------------------------------------------------

st.header("2. Target variable")
st.write("Churn = did not renew (`auto_renew = 0`). Renewed = `auto_renew = 1`.")
fig, ax = plt.subplots(figsize=(7, 4))
counts = features["auto_renew"].map({0: "Churned (auto_renew = 0)", 1: "Renewed (auto_renew = 1)"}).value_counts()
sns.barplot(x=counts.index, y=counts.values, ax=ax)
ax.set_title("Customers by outcome (auto_renew)")
ax.set_ylabel("Customers")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Held-out evaluation with the saved PySpark pipeline
# ---------------------------------------------------------------------------

st.header("3. Model evaluation (held-out test customers)")
st.write(
    f"Model: **{selected['selected_config']} / {selected['selected_model']}**, the PySpark pipeline saved in "
    f"`{selected['pipeline_path']}`. Metrics use only the 44 test customers that were never used for training."
)

# Re-score the test customers with the saved pipeline, so the dashboard shows the model rather than a cached file.
test_ids = preds["customer_id"].tolist()
try:
    spark = get_spark()
    model = load_pipeline(str(pipeline_path))
    sdf = spark.createDataFrame(features[features["customer_id"].isin(test_ids)].assign(churn_label=lambda d: (1 - d["auto_renew"]).astype(float)))
    scored = model.transform(sdf).select("customer_id", "churn_label", "prediction", "probability").toPandas()
    scored["churn_probability"] = scored["probability"].apply(lambda v: float(v[1]))
    scored = scored.drop(columns="probability")
    live = True
except Exception as exc:  # show the failure, do not hide it
    st.error(f"Could not run the saved pipeline: {exc}")
    scored = preds.rename(columns={"churn_label": "churn_label"})
    live = False

if live:
    merged = scored.merge(preds[["customer_id", "prediction"]], on="customer_id", suffixes=("", "_saved"))
    agree = int((merged["prediction"] == merged["prediction_saved"]).sum())
    st.caption(f"Live re-scoring agrees with the saved notebook predictions for {agree} of {len(merged)} test customers.")

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
m2.metric("Churn precision", f"{precision:.3f}")
m3.metric("Churn recall", f"{recall:.3f}")
m4.metric("Churn F1", f"{f1:.3f}")
m5.metric("ROC-AUC (churn)", f"{roc_auc:.3f}")
st.caption("Churn is the positive class. With 44 test customers, one customer changes precision or recall by several points.")

# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------

st.header("4. Confusion matrix")
cm = confusion_matrix(y_true, y_pred, labels=[RENEWED, CHURN])
fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Predicted renewed", "Predicted churned"],
            yticklabels=["Actual renewed", "Actual churned"])
ax.set_title("Held-out test customers")
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Classification report
# ---------------------------------------------------------------------------

st.header("5. Classification report")
report = classification_report(
    y_true, y_pred, labels=[RENEWED, CHURN], target_names=[LABEL_NAMES[RENEWED], LABEL_NAMES[CHURN]],
    output_dict=True, zero_division=0,
)
st.dataframe(pd.DataFrame(report).transpose().round(3))

# ---------------------------------------------------------------------------
# 6. ROC curve
# ---------------------------------------------------------------------------

st.header("6. ROC curve (churn)")
if len(np.unique(y_true)) == 2:
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=CHURN)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (churn recall)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Model comparison
# ---------------------------------------------------------------------------

st.header("7. Model comparison")
cut_results = load_results(OUTPUT_CUTOFF / "model_results.csv")
st.subheader("Fixed-cutoff features (use these)")
st.dataframe(cut_results[["config", "model", "cv_roc_auc", "roc_auc", "churn_precision", "churn_recall", "churn_f1", "accuracy"]].round(3))

fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(cut_results))
ax.bar(x - 0.2, cut_results["cv_roc_auc"], width=0.4, label="CV ROC-AUC")
ax.bar(x + 0.2, cut_results["roc_auc"], width=0.4, label="Test ROC-AUC")
ax.axhline(0.5, linestyle="--", color="grey", label="Chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(cut_results["config"] + "\n" + cut_results["model"], fontsize=8)
ax.set_ylim(0, 1)
ax.set_ylabel("ROC-AUC")
ax.legend(fontsize=8)
st.pyplot(fig)
plt.close(fig)

with st.expander("Original full-window results (leaky, for comparison only)"):
    st.warning("These used features that cover activity up to cancellation. Do not use them to judge renewal prediction.")
    canon_results = load_results(OUTPUT_CANON / "model_results.csv")
    st.dataframe(canon_results[["config", "model", "cv_roc_auc", "roc_auc", "churn_f1"]].round(3))

# ---------------------------------------------------------------------------
# 8. Prediction results
# ---------------------------------------------------------------------------

st.header("8. Test-customer predictions")
results = pd.DataFrame({
    "customer_id": scored["customer_id"].to_numpy(),
    "Actual": pd.Series(y_true).map(LABEL_NAMES).to_numpy(),
    "Predicted": pd.Series(y_pred).map(LABEL_NAMES).to_numpy(),
    "Correct": y_true == y_pred,
    "Churn probability": y_prob.round(4),
})
st.dataframe(results.sort_values("Churn probability", ascending=False), height=320)
st.download_button(
    "Download test predictions (CSV)",
    data=results.to_csv(index=False),
    file_name="test_predictions_cutoff14.csv",
    mime="text/csv",
)
