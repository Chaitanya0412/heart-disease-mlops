
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.joblib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("heart_api")

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found. Run python -m src.train first."
    )

model = joblib.load(MODEL_PATH)

REQUESTS = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)

LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds"
)

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description="Educational heart disease classifier"
)

app.mount("/metrics", make_asgi_app())


class Patient(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=20)
    slope: int = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=4)
    thal: int = Field(ge=3, le=7)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(patient: Patient):
    start = time.perf_counter()
    REQUESTS.inc()

    try:
        row = pd.DataFrame([patient.model_dump()])

        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]

        result = {
            "prediction": prediction,
            "label": (
                "disease_present"
                if prediction == 1
                else "disease_absent"
            ),
            "confidence": round(
                float(max(probabilities)), 4
            ),
            "probability_disease": round(
                float(probabilities[1]), 4
            )
        }

        logger.info(
            "Prediction completed; class=%s",
            prediction
        )

        return result

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    finally:
        LATENCY.observe(time.perf_counter() - start)
