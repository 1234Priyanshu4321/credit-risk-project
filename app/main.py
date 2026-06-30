from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.model import explain, load_artifacts, predict
from app.schema import ApplicantFeatures, ExplainResponse, PredictionResponse

_metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _metadata
    _metadata = load_artifacts()
    yield


app = FastAPI(
    title="Credit Risk API",
    description="Predicts credit default risk using a German Credit dataset–trained pipeline. "
                "Returns prediction, probability, and optional SHAP feature explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _metadata.get("best_model_name"),
        "cv_roc_auc": _metadata.get("cv_mean_roc_auc"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: ApplicantFeatures):
    try:
        result = predict(features.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/predict/explain", response_model=ExplainResponse)
def explain_endpoint(features: ApplicantFeatures, top_n: int = 10):
    try:
        result = explain(features.model_dump(), top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result
