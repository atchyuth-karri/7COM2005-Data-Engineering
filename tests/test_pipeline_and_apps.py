"""
Automated Verification & Smoke Test Suite for PerfectPetClub.
Module: 7COM2005 Data Engineering | University of Hertfordshire
Tests canonical datasets, model outputs, application modules, and reports.
"""

import os
import unittest
import py_compile
import pandas as pd
import sqlite3

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class TestPerfectPetClub(unittest.TestCase):

    def test_canonical_parquet_dataset(self):
        """Verifies canonical Parquet dataset dimensions and data contracts."""
        parquet_path = os.path.join(PROJECT_ROOT, "PerfectPetClub/data/processed/customer_churn_dataset.parquet")
        self.assertTrue(os.path.exists(parquet_path), "Canonical parquet file missing")
        df = pd.read_parquet(parquet_path)
        self.assertEqual(len(df), 221, "Dataset must have exactly 221 subscriber rows")
        self.assertEqual(len(df.columns), 36, "Dataset must have exactly 36 features")
        self.assertEqual(df.isnull().sum().sum(), 0, "Canonical feature store must have 0 missing values")
        self.assertEqual(df["customer_id"].nunique(), 221, "Customer IDs must be unique")

    def test_cutoff14_dataset(self):
        """Verifies 14-day observation horizon dataset."""
        p = os.path.join(PROJECT_ROOT, "PerfectPetClub/data/processed/customer_churn_dataset_cutoff14.parquet")
        self.assertTrue(os.path.exists(p), "Cutoff-14 parquet missing")
        df = pd.read_parquet(p)
        self.assertEqual(len(df), 221)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_cutoff21_dataset(self):
        """Verifies 21-day observation horizon dataset."""
        p = os.path.join(PROJECT_ROOT, "PerfectPetClub/data/processed/customer_churn_dataset_cutoff21.parquet")
        self.assertTrue(os.path.exists(p), "Cutoff-21 parquet missing")
        df = pd.read_parquet(p)
        self.assertEqual(len(df), 221)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_sqlite_database_connectivity(self):
        """Verifies read-only SQLite database integrity and table counts."""
        db_path = os.path.join(PROJECT_ROOT, "PerfectPetClub/data/raw/PerfectPetClub.db")
        self.assertTrue(os.path.exists(db_path), "SQLite DB missing")
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cursor.fetchall()]
            required_tables = ["customer", "subscription", "subscription_type", "sale", "sale_product", "product", "payment", "product_review"]
            for t in required_tables:
                self.assertIn(t, tables, f"Required table {t} missing from DB")

    def test_ml_results_csvs(self):
        """Verifies machine learning evaluation result tables."""
        for sub in ["ml", "ml_cutoff14", "ml_cutoff21"]:
            path = os.path.join(PROJECT_ROOT, f"PerfectPetClub/outputs/{sub}/model_results.csv")
            self.assertTrue(os.path.exists(path), f"Missing {path}")
            df = pd.read_csv(path)
            self.assertGreater(len(df), 0, f"Empty results in {path}")
            self.assertIn("cv_roc_auc", df.columns)
            self.assertIn("accuracy", df.columns)
            self.assertIn("roc_auc", df.columns)

    def test_python_scripts_syntax(self):
        """Verifies that all standalone application scripts compile cleanly."""
        scripts = [
            "app.py",
            "gata/app.py",
            "scripts/generate_all_results_report.py",
            "PerfectPetClub/scripts/download_survey_from_s3.py",
        ]
        for s in scripts:
            full_path = os.path.join(PROJECT_ROOT, s)
            self.assertTrue(os.path.exists(full_path), f"Script {s} missing")
            py_compile.compile(full_path, doraise=True)

    def test_master_all_results_html(self):
        """Verifies standalone master HTML report file."""
        html_path = os.path.join(PROJECT_ROOT, "reports/all_results_report.html")
        self.assertTrue(os.path.exists(html_path), "Master HTML report missing")
        self.assertGreater(os.path.getsize(html_path), 500_000, "Report file suspiciously small")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("🐾 PerfectPetClub", content)
        self.assertIn("data:image/png;base64,", content, "Embedded Base64 charts missing")
        self.assertIn("Atchyuth Karri", content)

if __name__ == "__main__":
    unittest.main()

