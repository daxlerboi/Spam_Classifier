import sys, io, json, hashlib, time
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
FRONTEND_DIR = BASE_DIR / "frontend"
BLACKLIST_PATH = BASE_DIR / "model" / "blacklist.json"
HONEYPOT_LOG_PATH = BASE_DIR / "model" / "honeypot_log.json"

sys.path.insert(0, str(BASE_DIR))
from src.preprocessing import clean_text

SPAM_THRESHOLD = 10
ip_violation_counts: dict[str, int] = defaultdict(int)
ip_last_violation: dict[str, float] = {}

def load_blacklist() -> dict:
    if BLACKLIST_PATH.exists():
        with open(BLACKLIST_PATH) as f:
            return json.load(f)
    return {}

def save_blacklist(data: dict):
    with open(BLACKLIST_PATH, "w") as f:
        json.dump(data, f, indent=2)

def load_honeypot_log() -> list:
    if HONEYPOT_LOG_PATH.exists():
        with open(HONEYPOT_LOG_PATH) as f:
            return json.load(f)
    return []

def save_honeypot_log(data: list):
    with open(HONEYPOT_LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

def record_honeypot_hit(ip: str, user_agent: str, body: dict):
    entry = {
        "timestamp": time.time(),
        "ip": ip,
        "user_agent": user_agent,
        "body": body,
    }
    log = load_honeypot_log()
    log.append(entry)
    save_honeypot_log(log)

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def is_ip_banned(ip: str) -> bool:
    now = time.time()
    if ip in ip_last_violation and (now - ip_last_violation[ip]) > 300:
        ip_violation_counts[ip] = 0
    return ip_violation_counts[ip] >= 10

def flag_ip(ip: str):
    ip_violation_counts[ip] += 1
    ip_last_violation[ip] = time.time()

print(" Loading model...")
try:
    model = joblib.load(str(MODEL_PATH))
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

app = FastAPI(
    title="Spam Classifier API",
    description="Predicts whether a message is SPAM or HAM (not spam).",
    version="2.0.0",
)

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

class FeedbackRequest(BaseModel):
    message: str = Field(..., description="The original message")
    correct_label: str = Field(..., description="Correct label: SPAM or HAM")
    predicted_label: str = Field(..., description="What the model predicted")

class FeedbackResponse(BaseModel):
    status: str
    blacklisted: bool
    message: str

class StatsResponse(BaseModel):
    status: str
    honeypot_hits: int
    blacklist_size: int
    banned_ips: int

@app.get("/health", response_model=HealthResponse)
@limiter.limit("60/minute")
def health(request: Request):

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

    return metrics

@app.get("/stats", response_model=StatsResponse)
def abuse_stats(request: Request):
    log = load_honeypot_log()
    blacklist = load_blacklist()
    unique_banned = sum(1 for count in ip_violation_counts.values() if count >= 10)
    return StatsResponse(
        status="running",
        honeypot_hits=len(log),
        blacklist_size=len(blacklist),
        banned_ips=unique_banned,
    )

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("10/minute")
def predict(req: PredictRequest, request: Request):
    ip = get_client_ip(request)
    if is_ip_banned(ip):
        return JSONResponse(
            status_code=403,
            content={"detail": "Too many abuse violations. IP temporarily blocked."},
        )

    cleaned = clean_text(req.message)
    msg_hash = hashlib.sha256(cleaned.encode()).hexdigest()
    blacklist = load_blacklist()

    if msg_hash in blacklist and blacklist[msg_hash].get("spam_count", 0) >= SPAM_THRESHOLD:
        flag_ip(ip)
        return JSONResponse(
            status_code=403,
            content={"detail": "Message blocked: identified as high-confidence spam."},
        )

    probs = model.predict_proba([cleaned])[0]
    pred = model.predict([cleaned])[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = float(max(probs))
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "spam_probability": round(float(probs[1]), 4),
        "ham_probability": round(float(probs[0]), 4),
    }

@app.post("/v1/predict")
@limiter.limit("60/minute")
def honeypot_predict(req: PredictRequest, request: Request):
    ip = get_client_ip(request)
    record_honeypot_hit(
        ip=ip,
        user_agent=request.headers.get("user-agent", "unknown"),
        body={"message": req.message},
    )
    print(f" [HONEYPOT] Hit from IP {ip}")
    return {
        "label": "HAM",
        "confidence": 0.95,
        "spam_probability": 0.05,
        "ham_probability": 0.95,
    }

@app.post("/feedback", response_model=FeedbackResponse)
@limiter.limit("30/minute")
def feedback(req: FeedbackRequest, request: Request):
    cleaned = clean_text(req.message)
    msg_hash = hashlib.sha256(cleaned.encode()).hexdigest()
    blacklist = load_blacklist()

    entry = blacklist.get(msg_hash, {"spam_count": 0, "ham_count": 0, "first_seen": time.time()})

    if req.correct_label == "SPAM":
        entry["spam_count"] = entry.get("spam_count", 0) + 1
    else:
        entry["ham_count"] = entry.get("ham_count", 0) + 1

    entry["last_updated"] = time.time()
    blacklist[msg_hash] = entry
    save_blacklist(blacklist)

    if entry["spam_count"] >= SPAM_THRESHOLD:
        return FeedbackResponse(
            status="blacklisted",
            blacklisted=True,
            message=f"Message blacklisted after {entry['spam_count']} spam reports.",
        )

    return FeedbackResponse(
        status="recorded",
        blacklisted=False,
        message=f"Feedback recorded. {SPAM_THRESHOLD - entry['spam_count']} more spam reports needed to blacklist.",
    )

@app.get("/", include_in_schema=False)
async def serve_root(request: Request):

    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(request: Request, path: str):

    file_path = FRONTEND_DIR / path
    if path and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    print(f"\n Server running on http://127.0.0.1:8000")
    print(f" UI → http://127.0.0.1:8000/")
    print(f" Docs → http://127.0.0.1:8000/docs")
    print(f" API → POST http://127.0.0.1:8000/predict")
    print(f" Honeypot → POST http://127.0.0.1:8000/v1/predict")
    uvicorn.run(app, host="127.0.0.1", port=8000)
