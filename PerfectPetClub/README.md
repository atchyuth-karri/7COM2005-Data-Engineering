# PerfectPetClub – Data Engineering & Analytics

**Module:** 7COM2005 – Data Engineering  
**Institution:** University of Hertfordshire  
**Project Type:** Team Project  

---

## Project Overview

**PerfectPetClub** is a comprehensive data engineering and machine learning project focused on designing end-to-end data pipelines for pet subscription and e-commerce services. The solution extracts and ingests data from heterogeneous sources, performs scalable transformation and data cleaning, and builds analytical models to derive actionable business insights.

---

## Directory Structure

```text
PerfectPetClub/
│
├── data/
│   ├── raw/
│   │   ├── PerfectPetClub.db
│   │   ├── PerfectPetClub_CRM_report.csv
│   │   ├── PerfectPetClub_server_log.csv
│   │   └── PerfectPetClub_survey.csv
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_ETL_Pipeline.ipynb
│   └── 02_Machine_Learning.ipynb
│
├── documentation/
│
├── requirements.txt
└── README.md
```

---

## Data Sources

The raw data sources are housed in `data/raw/`:

1. **`PerfectPetClub.db`** (SQLite):
   - Operational database containing transactional tables: `customer`, `product`, `sale`, `sale_product`, `subscription`, `subscription_type`, `payment`, and `product_review`.
2. **`PerfectPetClub_CRM_report.csv`**:
   - Customer relationship management logs capturing support queries and complaints across multiple communication channels (`e-mail`, `live-chat`, `telephone`).
3. **`PerfectPetClub_server_log.csv`**:
   - Web server access logs detailing user browsing behavior, sessions, IP addresses, timestamps, and client user-agents.
4. **`PerfectPetClub_survey.csv`**:
   - Customer satisfaction feedback and survey ratings.

---

## Notebooks Workflow

- **`notebooks/01_ETL_Pipeline.ipynb`**:
  - Connects to SQLite database and loads external CSV logs.
  - Cleans missing values, handles duplicates, parses dates, and harmonizes customer identifiers.
  - Generates unified customer interaction datasets into `data/processed/`.

- **`notebooks/02_Machine_Learning.ipynb`**:
  - Ingests processed data from `data/processed/`.
  - Conducts exploratory data analysis (EDA) and feature engineering.
  - Trains, tunes, and evaluates machine learning models (e.g., customer churn prediction, subscription renewal likelihood, or customer lifetime value segmentation).

---

## Getting Started

### 1. Environment Setup

Activate the project virtual environment and install the required dependencies:

```bash
# Activate virtual environment
source ../.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

