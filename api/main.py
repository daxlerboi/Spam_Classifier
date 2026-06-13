"""
FastAPI Backend — Spam Classifier API

Endpoints:
  GET  /          → web UI (index.html)
  GET  /health    → health check + model info
  POST /predict   → classify a message
  GET  /metrics   → model performance metrics
  GET  /docs      → Swagger API docs (built-in)
"""
import sys, io, json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import joblib

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "model" / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
FRONTEND_DIR = BASE_DIR / "frontend"

# Add project root to path so we can import src.preprocessing
sys.path.insert(0, str(BASE_DIR))
from src.preprocessing import clean_text

# ── Load model ───────────────────────────────────────────────────────
print(" Loading model...")
try:
    model   = joblib.load(str(MODEL_PATH))
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    n_features = metrics.get("n_features", "?")
    accuracy = metrics.get("accuracy", 0)
    f1 = metrics.get("f1", 0)
    print(f" Model loaded — {n_features} features, F1: {f1:.1%}")

except FileNotFoundError as e:
    print(f" ERROR: {e}")
    print(" Run 'python -m src.train' first to create the model.")
    sys.exit(1)
except Exception as e:
    print(f" ERROR loading model: {e}")
    sys.exit(1)

# ── App setup ────────────────────────────────────────────────────────
app = FastAPI(
    title="Spam Classifier API",
    description="Predicts whether a message is SPAM or HAM (not spam).",
    version="2.0.0",
)

# Rate limiting setup
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The message text to classify",
    )

class PredictResponse(BaseModel):
    label: str
    confidence: float
    spam_probability: float
    ham_probability: float

class HealthResponse(BaseModel):
    status: str
    model: str
    features: int
    f1: str
    accuracy: str
    n_samples: int

# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
def health(request: Request):
    """JSON health check for developers."""
    return {
        "status": "running",
        "model": "TF-IDF + MultinomialNB",
        "features": metrics["n_features"],
        "f1": f"{metrics.get('f1', 0):.2%}",
        "accuracy": f"{metrics['accuracy']:.2%}",
        "n_samples": metrics["n_samples"],
    }

@app.get("/metrics")
@limiter.limit("60/minute")
def get_metrics(request: Request):
    """Full performance metrics."""
    return metrics

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("10/minute")
def predict(req: PredictRequest, request: Request):
    """Classify a message as SPAM or HAM."""
    cleaned = clean_text(req.message)
    probs = model.predict_proba([cleaned])[0]   # [P(ham), P(spam)]
    pred  = model.predict([cleaned])[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = float(max(probs))
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "spam_probability": round(float(probs[1]), 4),
        "ham_probability":  round(float(probs[0]), 4),
    }

# ── Serve frontend ──────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_root(request: Request):
    """Serve the web UI at the root URL."""
    # For the root path, serve index.html (the JS handles API calls relative to /)
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(request: Request, path: str):
    """Serve static files from the frontend folder (fallback to index.html for SPA)."""
    file_path = FRONTEND_DIR / path
    if path and file_path.is_file():
        return FileResponse(file_path)
    # Fallback: serve the main page (for direct navigation)
    return FileResponse(FRONTEND_DIR / "index.html")

# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"\n Server running on http://127.0.0.1:8000")
    print(f" UI    → http://127.0.0.1:8000/")
    print(f" Docs  → http://127.0.0.1:8000/docs")
    print(f" API   → POST http://127.0.0.1:8000/predict")
    uvicorn.run(app, host="127.0.0.1", port=8000)
