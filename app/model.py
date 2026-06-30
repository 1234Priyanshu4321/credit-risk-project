"""
Loads the trained pipeline at startup.
Exposes predict() and explain() — called by routes.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts"

_pipeline = None
_metadata = None
_explainer = None


def load_artifacts():
    global _pipeline, _metadata, _explainer
    _pipeline = joblib.load(ARTIFACT_DIR / "model.joblib")
    _metadata = joblib.load(ARTIFACT_DIR / "metadata.joblib")

    clf = _pipeline.named_steps["clf"]
    model_name = _metadata["best_model_name"]

    if model_name in ("Random Forest", "XGBoost"):
        _explainer = shap.TreeExplainer(clf)
    elif model_name == "Logistic Regression":
        preprocessor = _pipeline.named_steps["preprocessor"]
        # LinearExplainer needs background — use zero vector (mean-centered by StandardScaler, so zeros ≈ mean)
        n_features = preprocessor.transform(
            pd.DataFrame([{c: 0 for c in preprocessor.feature_names_in_}])
        ).shape[1]
        _explainer = shap.LinearExplainer(clf, masker=shap.maskers.Independent(np.zeros((1, n_features))))

    return _metadata


def _to_frame(features_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([features_dict])


def predict(features_dict: dict) -> dict:
    df = _to_frame(features_dict)
    prob = float(_pipeline.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    return {
        "prediction": pred,
        "probability_high_risk": round(prob, 4),
        "model_used": _metadata["best_model_name"],
    }


def explain(features_dict: dict, top_n: int = 10) -> dict:
    df = _to_frame(features_dict)
    X_transformed = _pipeline.named_steps["preprocessor"].transform(df)
    feature_names = _pipeline.named_steps["preprocessor"].get_feature_names_out()

    shap_values = _explainer.shap_values(X_transformed)

    # For binary tree models shap_values may be list[2] — take class-1 values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    sv = np.array(shap_values)
    # TreeExplainer for multiclass/binary RF: shape (n_samples, n_features, n_classes) → take class-1
    if sv.ndim == 3:
        row = sv[0, :, 1]
    elif sv.ndim == 2:
        row = sv[0]
    else:
        row = sv

    contributions = sorted(
        [
            {"feature": name, "shap_value": round(float(val), 5)}
            for name, val in zip(feature_names, row)
        ],
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )[:top_n]

    prob = float(_pipeline.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability_high_risk": round(prob, 4),
        "model_used": _metadata["best_model_name"],
        "top_features": contributions,
    }
