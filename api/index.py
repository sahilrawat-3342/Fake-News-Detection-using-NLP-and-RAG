"""
TruthLens REST API.

Exposes Layer 1 (TF-IDF + Logistic Regression) and Layer 2 (live RAG)
endpoints for the React frontend, compatible with both local development
and Vercel Serverless Functions.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory and current directory are on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

for p in (ROOT_DIR, CURRENT_DIR, Path.cwd()):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

MAX_INPUT_CHARS = 10_000

app = FastAPI(title="TruthLens API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _find_file(rel_path: str) -> Optional[Path]:
    """Search multiple candidate roots (local workspace, Vercel /var/task)."""
    candidates = [
        ROOT_DIR / rel_path,
        CURRENT_DIR / rel_path,
        Path.cwd() / rel_path,
        Path("/var/task") / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


MODEL_PATH = _find_file("models/final_model.pkl") or (ROOT_DIR / "models" / "final_model.pkl")
METRICS_PATH = _find_file("performance_metrics.json") or (ROOT_DIR / "performance_metrics.json")
DATA_DIR = _find_file("data") or (ROOT_DIR / "data")

_model_lock = threading.Lock()
_model_cache: Dict[str, Any] = {"model": None, "mtime": None, "error": None}
_runtime_state: Dict[str, Any] = {"train_time": None, "data_count": 44898}


class TextPayload(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_INPUT_CHARS)


def _load_model():
    """Lazy-load the persisted sklearn pipeline with defensive error handling."""
    if not MODEL_PATH.exists():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    with _model_lock:
        if _model_cache["model"] is None or _model_cache["mtime"] != mtime:
            try:
                import joblib

                _model_cache["model"] = joblib.load(MODEL_PATH)
                _model_cache["mtime"] = mtime
                _model_cache["error"] = None
            except Exception as exc:
                _model_cache["error"] = str(exc)
                return None
        return _model_cache["model"]


def _load_metrics() -> Dict[str, Any]:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "models": {
            "truthlens_layer1": {
                "model_name": "TruthLens Layer 1",
                "algorithm": "Logistic Regression + TF-IDF",
                "metrics": {
                    "accuracy": 0.9852538170429336,
                    "precision": 0.9806375442739079,
                    "recall": 0.9925908221797323,
                    "f1_score": 0.9865779783822307,
                    "roc_auc": 0.9985399971530906,
                    "confusion_matrix": {
                        "true_negatives": 3397,
                        "false_positives": 82,
                        "false_negatives": 31,
                        "true_positives": 4153,
                    },
                },
            }
        }
    }


def _clean_text_fallback(text: str) -> str:
    """Safe fallback text cleaning mirroring src.data_loader."""
    try:
        from src.data_loader import DataProcessor

        return DataProcessor().clean_text(text)
    except Exception:
        import re

        t = str(text).lower()
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
        t = re.sub(r"<.*?>", " ", t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()


def _has_env(*keys: str) -> bool:
    return any(os.getenv(k) for k in keys)


@app.get("/api/health")
def health():
    model = _load_model()
    return {
        "status": "ok",
        "model_online": model is not None,
        "model_file_present": MODEL_PATH.exists(),
        "model_error": _model_cache.get("error"),
        "rag_configured": _has_env("GROQ_API_KEY")
        and _has_env("SERPAPI_KEY", "SERPAPI_API_KEY", "GOOGLE_API_KEY"),
        "llm_model": os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant",
        "retrain_available": (DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists(),
        "is_vercel": bool(os.getenv("VERCEL")),
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
        err_msg = _model_cache.get("error") or "Model artifact offline (models/final_model.pkl missing)."
        raise HTTPException(status_code=503, detail=f"Model unavailable: {err_msg}")

    clean_input = _clean_text_fallback(text)
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
        from src.rag_engine import (
            RAGConfigurationError,
            build_groq_llm,
            extract_claim_and_query,
            search_web,
            verify_claim_with_context,
        )

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
        return {
            "claim": extraction["claim"],
            "query": extraction["query"],
            "extraction_parse_ok": extraction["parse_ok"],
            "search_results": search_results,
            "elapsed_s": round(elapsed, 2),
            **verification,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Deep verification failed: {exc}") from exc


@app.post("/api/retrain")
def retrain():
    if not ((DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists()):
        raise HTTPException(
            status_code=501,
            detail="Training datasets are not deployed in this environment. Retrain locally.",
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
