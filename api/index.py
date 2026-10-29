"""
TruthLens REST API.

Exposes Layer 1 (TF-IDF + Logistic Regression) and Layer 2 (live RAG)
endpoints, and serves the compiled React + Playful Motion frontend bundle.
Compatible with both local development and Vercel Serverless Functions.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def vercel_routing_middleware(request: Request, call_next):
    """Normalize Vercel Edge rewrites and path resolution."""
    matched = request.headers.get("x-matched-path")
    if matched:
        raw_path = matched.split("?")[0]
        if raw_path and not raw_path.startswith("/api/index"):
            request.scope["path"] = raw_path

    cur_path = request.scope.get("path", "")
    for prefix in ("/api/index.py", "/api/index"):
        if cur_path == prefix:
            request.scope["path"] = "/"
            break
        elif cur_path.startswith(prefix + "/"):
            request.scope["path"] = cur_path[len(prefix):]
            break

    return await call_next(request)


# --- LOCATE ASSETS & PERSISTED ARTIFACTS ---

def _find_dir_or_file(rel_path: str) -> Optional[Path]:
    """Search candidate roots across local workspace and Vercel serverless."""
    candidates = [
        ROOT_DIR / rel_path,
        CURRENT_DIR / rel_path,
        Path.cwd() / rel_path,
        Path("/var/task") / rel_path,
        Path("/var/task/api") / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_dist_dir() -> Optional[Path]:
    for candidate in [
        CURRENT_DIR / "dist",
        ROOT_DIR / "dist",
        CURRENT_DIR.parent / "dist",
        Path("/var/task/dist"),
        Path("/var/task/api/dist"),
        Path.cwd() / "dist",
        Path.cwd() / "api" / "dist",
    ]:
        if candidate.exists() and (candidate / "index.html").exists():
            return candidate
    return None


MODEL_PATH = _find_dir_or_file("models/final_model.pkl") or (ROOT_DIR / "models" / "final_model.pkl")
METRICS_PATH = _find_dir_or_file("performance_metrics.json") or (ROOT_DIR / "performance_metrics.json")
DATA_DIR = _find_dir_or_file("data") or (ROOT_DIR / "data")
DIST_DIR = _find_dist_dir()

# Cache static frontend files in memory for ~0ms latency and zero disk contention
_INDEX_HTML: Optional[bytes] = None
_FAVICON_SVG: Optional[bytes] = None
_ASSET_CACHE: Dict[str, bytes] = {}

if DIST_DIR:
    try:
        idx_file = DIST_DIR / "index.html"
        if idx_file.exists():
            _INDEX_HTML = idx_file.read_bytes()
        fav_file = DIST_DIR / "favicon.svg"
        if fav_file.exists():
            _FAVICON_SVG = fav_file.read_bytes()
        assets_dir = DIST_DIR / "assets"
        if assets_dir.exists():
            for item in assets_dir.glob("*"):
                if item.is_file():
                    _ASSET_CACHE[item.name] = item.read_bytes()
    except Exception:
        pass

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


# --- API ROUTE HANDLERS ---

def handle_health():
    model = _load_model()
    return {
        "status": "ok",
        "model_online": model is not None,
        "model_file_present": MODEL_PATH.exists(),
        "model_error": _model_cache.get("error"),
        "dist_present": _INDEX_HTML is not None,
        "rag_configured": _has_env("GROQ_API_KEY")
        and _has_env("SERPAPI_KEY", "SERPAPI_API_KEY", "GOOGLE_API_KEY"),
        "llm_model": os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant",
        "retrain_available": (DATA_DIR / "Fake.csv").exists() and (DATA_DIR / "True.csv").exists(),
        "is_vercel": bool(os.getenv("VERCEL")),
    }


def handle_metrics():
    data = _load_metrics()
    models = data.get("models", {})
    return {
        "models": models,
        "layer1": models.get("truthlens_layer1", {}).get("metrics", {}),
        "data_count": _runtime_state["data_count"],
        "train_time": _runtime_state["train_time"],
    }


def handle_predict(payload: TextPayload):
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


def handle_verify(payload: TextPayload):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input payload is empty.")

    try:
        from src.rag_engine import (
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


def handle_retrain():
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


# --- STATIC ASSET SERVING ---

def serve_favicon():
    if _FAVICON_SVG is not None:
        return Response(
            content=_FAVICON_SVG,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    raise HTTPException(status_code=404, detail="Favicon not found")


def serve_asset(filename: str):
    content = _ASSET_CACHE.get(filename)
    if content is None and DIST_DIR:
        f = DIST_DIR / "assets" / filename
        if f.exists() and f.is_file():
            content = f.read_bytes()
            _ASSET_CACHE[filename] = content

    if content is None:
        raise HTTPException(status_code=404, detail=f"Asset '{filename}' not found")

    media_type = "application/octet-stream"
    if filename.endswith(".js"):
        media_type = "application/javascript"
    elif filename.endswith(".css"):
        media_type = "text/css"
    elif filename.endswith(".svg"):
        media_type = "image/svg+xml"
    elif filename.endswith(".woff2"):
        media_type = "font/woff2"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


def serve_index_html():
    if _INDEX_HTML is not None:
        return Response(
            content=_INDEX_HTML,
            media_type="text/html; charset=utf-8",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )
    return Response(
        content="""<!doctype html><html><head><title>TruthLens</title></head>
        <body style="font-family:sans-serif;padding:3rem;background:#faf8f5;">
        <h1>TruthLens API</h1><p>Frontend bundle is initializing. Visit <a href="/docs">/docs</a> for API.</p>
        </body></html>""",
        media_type="text/html; charset=utf-8",
    )


def serve_spa_fallback(full_path: str):
    # Do not swallow API calls or swagger docs
    if full_path.startswith("api/") or full_path in ("docs", "openapi.json", "redoc"):
        raise HTTPException(status_code=404, detail="Not Found")
    return serve_index_html()


# --- REGISTER ROUTES ---

# API endpoints
app.get("/api/health")(handle_health)
app.get("/health")(handle_health)
app.get("/api/metrics")(handle_metrics)
app.get("/metrics")(handle_metrics)
app.post("/api/predict")(handle_predict)
app.post("/predict")(handle_predict)
app.post("/api/verify")(handle_verify)
app.post("/verify")(handle_verify)
app.post("/api/retrain")(handle_retrain)
app.post("/retrain")(handle_retrain)

# Assets & Static
app.get("/favicon.svg")(serve_favicon)
app.get("/assets/{filename:path}")(serve_asset)

# Root & SPA catch-all
app.get("/")(serve_index_html)
app.get("/api/index.py")(serve_index_html)
app.get("/{full_path:path}")(serve_spa_fallback)
