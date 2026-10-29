"""
TruthLens REST API.

Exposes the existing Layer 1 (TF-IDF + Logistic Regression) classifier and the
Layer 2 (live RAG) verifier to the React frontend.

Run locally:
    uvicorn api.index:app --reload --port 8000
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import DataProcessor  # noqa: E402
from src.rag_engine import (  # noqa: E402
    DEFAULT_GROQ_MODEL,
    RAGConfigurationError,
    build_groq_llm,
    extract_claim_and_query,
    search_web,
    verify_claim_with_context,
)

MODEL_PATH = ROOT_DIR / "models" / "final_model.pkl"
METRICS_PATH = ROOT_DIR / "performance_metrics.json"
DATA_DIR = ROOT_DIR / "data"
MAX_INPUT_CHARS = 10_000

app = FastAPI(title="TruthLens API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_processor = DataProcessor()
_model_lock = threading.Lock()
_model_cache: Dict[str, Any] = {"model": None, "mtime": None}
_runtime_state: Dict[str, Any] = {"train_time": None, "data_count": 44898}


class TextPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


def _load_model():
    """Load (and hot-reload after retraining) the persisted sklearn pipeline."""
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    with _model_lock:
        if _model_cache["model"] is None or _model_cache["mtime"] != mtime:
            _model_cache["model"] = joblib.load(MODEL_PATH)
            _model_cache["mtime"] = mtime
        return _model_cache["model"]


def _load_metrics() -> Dict[str, Any]:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _has_env(*keys: str) -> bool:
    return any(os.getenv(k) for k in keys)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_online": MODEL_PATH.exists(),
        "rag_configured": _has_env("GROQ_API_KEY")
        and _has_env("SERPAPI_KEY", "SERPAPI_API_KEY", "GOOGLE_API_KEY"),
        "llm_model": os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
        "retrain_available": (DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists(),
    }


@app.get("/api/metrics")
def metrics():
    data = _load_metrics()
    models = data.get("models", {})
    return {
        "models": models,
        "layer1": models.get("truthlens_layer1", {}).get("metrics", {}),
        "data_count": _runtime_state["data_count"],
        "train_time": _runtime_state["train_time"],
    }


@app.post("/api/predict")
def predict(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input payload is empty.")

    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact offline (models/final_model.pkl missing).")

    clean_input = _processor.clean_text(text)
    t0 = time.perf_counter()
    pred = int(model.predict([clean_input])[0])
    real_prob: Optional[float] = None
    fake_prob: Optional[float] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([clean_input])[0]
        fake_prob = float(proba[0] * 100)
        real_prob = float(proba[1] * 100)
    latency_ms = (time.perf_counter() - t0) * 1000

    confidence = max(real_prob, fake_prob) if real_prob is not None else None
    return {
        "label": "real" if pred == 1 else "fake",
        "prediction": pred,
        "confidence": confidence,
        "real_probability": real_prob,
        "fake_probability": fake_prob,
        "latency_ms": round(latency_ms, 2),
        "clean_text": clean_input[:1500],
        "token_count": len(clean_input.split()),
    }


@app.post("/api/verify")
def verify(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input payload is empty.")

    try:
        t0 = time.perf_counter()
        llm = build_groq_llm()
        extraction = extract_claim_and_query(text, llm=llm)
        search_results = search_web(query=extraction["query"], max_results=5, enrich=True)
        verification = verify_claim_with_context(
            claim_text=extraction["claim"],
            search_results=search_results,
            raw_text=text,
            llm=llm,
        )
        elapsed = time.perf_counter() - t0
    except RAGConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Deep verification failed: {exc}") from exc

    return {
        "claim": extraction["claim"],
        "query": extraction["query"],
        "extraction_parse_ok": extraction["parse_ok"],
        "search_results": search_results,
        "elapsed_s": round(elapsed, 2),
        **verification,
    }


@app.post("/api/retrain")
def retrain():
    if not ((DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists()):
        raise HTTPException(
            status_code=501,
            detail="Training datasets are not available in this deployment. Retrain locally.",
        )

    from train_model import train_and_save_model

    cwd = os.getcwd()
    try:
        os.chdir(ROOT_DIR)
        start = time.perf_counter()
        acc, count = train_and_save_model()
        elapsed = time.perf_counter() - start
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}") from exc
    finally:
        os.chdir(cwd)

    _runtime_state["train_time"] = f"{elapsed:.2f}s"
    _runtime_state["data_count"] = int(count)
    return {"accuracy": float(acc), "data_count": int(count), "train_time": _runtime_state["train_time"]}
