"""
Training script for German Credit Risk model.
Mirrors credit_risk_model.ipynb logic, adds persistence.

Run: python train.py
Output: artifacts/model.joblib, artifacts/columns.joblib, artifacts/metadata.joblib
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ARTIFACT_DIR = "artifacts"

COLUMN_NAMES = [
    "Status", "Duration", "CreditHistory", "Purpose",
    "CreditAmount", "Savings", "EmploymentDuration",
    "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
    "ResidenceDuration", "Property", "Age",
    "OtherInstallmentPlans", "Housing", "ExistingCredits",
    "Job", "NumberOfDependents", "Telephone",
    "ForeignWorker", "Risk",
]


def load_data(path="german.data"):
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMN_NAMES)
    df["Risk"] = df["Risk"].map({1: 0, 2: 1})
    y = df["Risk"]
    X = df.drop(columns=["Risk"])
    return X, y


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include="int64").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])


from sklearn.base import clone


def build_pipelines(preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss", use_label_encoder=False,
            )),
        ]),
    }


def select_best_by_cv(pipelines, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_means, cv_stds = {}, {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        cv_means[name] = scores.mean()
        cv_stds[name] = scores.std()
        print(f"{name}: mean ROC-AUC={scores.mean():.4f} std={scores.std():.4f}")
    best_name = max(cv_means, key=cv_means.get)
    return best_name, cv_means, cv_stds


def main():
    X, y = load_data()
    columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    pipelines = build_pipelines(preprocessor)
    best_name, cv_means, cv_stds = select_best_by_cv(pipelines, X, y)
    print(f"\nBest Model Selected (by CV mean ROC-AUC): {best_name}")

    # Fit best model on full dataset for the deployed artifact.
    best_pipeline = build_pipelines(preprocessor)[best_name]
    best_pipeline.fit(X, y)

    # Held-out test ROC-AUC for reporting (separate fit on train split only).
    eval_pipeline = build_pipelines(preprocessor)[best_name]
    eval_pipeline.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, eval_pipeline.predict_proba(X_test)[:, 1])
    print(f"Held-out test ROC-AUC ({best_name}): {test_auc:.4f}")

    import os
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(best_pipeline, f"{ARTIFACT_DIR}/model.joblib")
    joblib.dump(columns, f"{ARTIFACT_DIR}/columns.joblib")

    metadata = {
        "best_model_name": best_name,
        "cv_mean_roc_auc": cv_means,
        "cv_std_roc_auc": cv_stds,
        "held_out_test_roc_auc": test_auc,
        "n_features": len(columns),
    }
    joblib.dump(metadata, f"{ARTIFACT_DIR}/metadata.joblib")
    with open(f"{ARTIFACT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=float)

    print(f"\nSaved: {ARTIFACT_DIR}/model.joblib, columns.joblib, metadata.joblib")


if __name__ == "__main__":
    main()
