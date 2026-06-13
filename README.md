# Spam Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-000000?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Built by `daxler_boi`

Try it live: [https://spam-classifier-dvfd.onrender.com/](https://spam-classifier-dvfd.onrender.com/)

Spam Classifier is a small Python machine learning project that detects whether an incoming text message is **spam** or **ham** (legitimate). It trains a model on real SMS data, saves the trained pipeline, and serves predictions through a lightweight web API with a dark-mode frontend.

## What It Does

The project does a few focused things:

- loads a real SMS spam dataset (5,572 messages from UCI)
- cleans and preprocesses the raw text (lowercase, strip URLs, phones, emails, punctuation)
- converts text into numeric features using TF-IDF (word importance scoring)
- trains a Multinomial Naive Bayes classifier with 5-fold cross-validation
- evaluates accuracy, precision, recall, and F1 on a held-out test set
- saves the trained pipeline and metrics to disk
- serves predictions through a FastAPI web server with a vanilla JS frontend

## How It Is Built

This project follows a modular ML pipeline split across four main files:

- **`src/preprocessing.py`** — text cleaning and data loading from CSV/TSV files
- **`src/train.py`** — orchestrates the full training pipeline: load, clean, vectorize, train, evaluate, save
- **`api/main.py`** — loads the saved model and exposes it through REST endpoints
- **`frontend/index.html`** — a single-file dark-mode web UI that calls the API with `fetch()`

The model uses TF-IDF vectorization (unigrams + bigrams, 5,000 max features) fed into a Multinomial Naive Bayes classifier with Laplace smoothing. No frontend build tools needed — just open the HTML file in a browser.

## Architecture

The project is split into three layers: **data**, **model**, and **serving**. Each layer is independent — you can retrain the model without touching the API, swap the frontend without retraining anything, or replace the dataset with your own.

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                            │
│                   frontend/index.html                            │
│                                                                  │
│  Vanilla JS — textarea for input, results display with          │
│  probability bars, example chips, fetches from /health and      │
│  /predict. No build step, no framework.                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (same origin)
                           │ POST /predict  {"message": "..."}
                           │ GET  /health    → model stats
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                                 │
│                   api/main.py (FastAPI)                          │
│                                                                  │
│  Loads model + metrics at startup (one-time, in-memory).        │
│  Exposes endpoints: /health, /predict, /metrics, / (frontend).  │
│  CORS enabled. Validates input with Pydantic schemas.           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ pipeline.predict_proba()
                           │ pipeline.predict()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MODEL LAYER                              │
│                   model/spam_model.pkl                           │
│                                                                  │
│  A scikit-learn Pipeline saved with joblib:                    │
│    Step 1: TfidfVectorizer — turns text into numeric vectors    │
│    Step 2: MultinomialNB — classifies vectors as spam/ham      │
│                                                                  │
│  Only one file needed — it contains both the vectorizer        │
│  and the classifier. Loading it restores the full pipeline.     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ was trained by
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                              │
│                   src/train.py                                   │
│                                                                  │
│  Orchestrates the full ML pipeline. Runs once to produce        │
│  the saved model. Not part of the live server.                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ reads data from
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│                   src/preprocessing.py + data/spam.csv           │
│                                                                  │
│  preprocessing.py: two functions                              │
│    clean_text()   — regex-based text cleaning                  │
│    load_csv()     — reads TSV, detects delimiter, strips        │
│                      header, returns (texts, labels)            │
│                                                                  │
│  data/spam.csv: 5,572 TSV rows (label TAB message).            │
│  UCI SMS Spam Collection — 747 spam, 4825 ham.                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow (Training)

```
Training run starts
        │
        ▼
load_csv("data/spam.csv")
  → reads 5,572 TSV rows
  → detects tab delimiter, no header
  → returns texts=[...], labels=[0,1,0,1...]
        │
        ▼
clean_text() applied to every message
  → lowercase, strip URLs/phones/emails/punctuation
  → "Win FREE money!!!" becomes "win free money"
        │
        ▼
train_test_split(texts, labels, stratify=labels)
  → 4,457 messages for training (80%)
  → 1,115 messages for testing  (20%)
  → both splits keep 13.4% spam ratio
        │
        ▼
Pipeline.fit(X_train, y_train)
  → TfidfVectorizer learns 5,000 word features from training text
  → MultinomialNB learns probability patterns from those features
        │
        ▼
Cross-validation (5 folds on training set)
  → F1 scores: [0.932, 0.918, 0.940, 0.905, 0.934]
  → Mean: 0.926, Std: 0.013 → model is stable
        │
        ▼
Pipeline.predict(X_test)
  → Accuracy 98.21%, Precision 97.78%, Recall 88.59%
  → Confusion matrix: 963 true ham, 132 true spam caught
        │
        ▼
Save artifacts
  → model/spam_model.pkl (full pipeline, ~5MB)
  → model/metrics.json   (scores for the API to read)
```

### Data Flow (Prediction)

```
User types message in browser textarea
        │
        ▼
JavaScript: fetch("POST /predict", {"message": "..."})
        │
        ▼
FastAPI: predict() endpoint
  → Pydantic validates: message is a non-empty string
        │
        ▼
pipeline.predict_proba([message])
  → vectorizer.transform() converts text to TF-IDF vector
  → MultinomialNB.predict_proba() returns [P(ham), P(spam)]
  → e.g. [0.04, 0.96] for a clear spam message
        │
        ▼
FastAPI returns JSON
  → {"label": "SPAM", "confidence": 0.96,
     "spam_probability": 0.96, "ham_probability": 0.04}
        │
        ▼
JavaScript updates the DOM
  → colors result box (red/green)
  → animates probability bar widths
  → shows confidence percentage
```

### Layer Responsibilities

| Layer | Runs During | Responsibility | Output |
|-------|------------|----------------|--------|
| **Preprocessing** | Training | Clean raw text, load CSV/TSV | Cleaned texts + labels |
| **Training** | Once (on demand) | Train, evaluate, save model | `spam_model.pkl`, `metrics.json` |
| **API** | Server runtime | Load model, serve predictions | HTTP responses |
| **Frontend** | Browser | User input UI, display results | Visual predictions |

### Why These Layers

The training layer and the serving layer are completely separate. This means:
- You can **retrain** the model at any time by re-running `python -m src.train` without changing any API code — the API reads whatever model file is on disk.
- You can **swap the frontend** for a mobile app, CLI tool, or Slack bot — as long as it sends `POST /predict` with the right JSON, nothing else needs to change.
- You can **replace the model** — swap MultinomialNB for Logistic Regression, XGBoost, or a neural net. As long as the new model implements `.predict()` and `.predict_proba()`, the API and frontend need zero changes.

## Code Walkthrough

This file has two functions that prepare raw text for the model.

**`clean_text(text)`** — takes a raw string and runs it through a 4-step cleaning pipeline:

```python
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Step 1: lowercase everything
    text = text.lower()

    # Step 2: remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 3: remove phone numbers (7-15 digits with optional dashes/spaces and leading '+')
    text = re.sub(r"(?<!\w)\+?[\d\s\-]{7,15}(?!\w)", " ", text)

    # Step 4: remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Step 5: remove all punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 6: collapse multiple spaces into one and strip edges
    text = re.sub(r"\s+", " ", text).strip()

    return text
```

The regex patterns handle the most common spammy elements:
- `https?://\S+|www\.\S+` catches both `http://` and `https://` links, plus bare `www.` domains
- `(?<!\w)\+?[\d\s\-]{7,15}(?!\w)` catches international phone numbers like "08452810075", "+44 20 7946 0958", or "0871 21-21-21"
- `\S+@\S+` catches any `user@domain.com` pattern

**`load_csv(path)`** — reads the dataset file and returns two lists: `texts` (clean messages) and `labels` (0 for ham, 1 for spam):

```python
def load_csv(path: str) -> tuple:
    import csv

    texts, labels = [], []

    # Read the first line to figure out the file format
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()

    # The SMS dataset uses tabs as separators
    if "\t" in first_line:
        delimiter = "\t"
    else:
        delimiter = ","

    # Heuristic: if the first field is "ham" or "spam", there is no header row
    first_field = first_line.split(delimiter)[0].strip().lower()
    has_header = first_field not in ("ham", "spam")

    # Read the actual data
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)  # skip the header row

        for row in reader:
            if len(row) < 2:
                continue  # skip malformed lines

            label_raw = row[0].strip().lower()  # "ham" or "spam"
            text_raw  = row[1].strip()          # the actual message

            label = 1 if label_raw == "spam" else 0  # 1 = spam, 0 = ham

            if text_raw:
                texts.append(clean_text(text_raw))  # clean the text first
                labels.append(label)

    return texts, labels
```

The `errors="replace"` in the `open()` call handles any weird characters in the dataset by replacing them with a safe placeholder instead of crashing.

---

### `src/train.py` — Training Pipeline

This is the main script. It loads data, trains the model, evaluates it, and saves everything. Here is what each section does:

**1. Setup and data loading:**

```python
BASE_DIR   = Path(__file__).resolve().parent.parent  # project root
DATA_PATH  = BASE_DIR / "data" / "spam.csv"
MODEL_DIR  = BASE_DIR / "model"

texts, labels = load_csv(DATA_PATH)
labels = np.array(labels)
```

Using `Path(__file__)` instead of a hardcoded path means the script works no matter where you run it from.

**2. Train/test split:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,        # 20% of data reserved for testing
    random_state=42,      # reproducible split every time
    stratify=labels,      # keep the same spam/ham ratio in both sets
)
```

`stratify=labels` is important — without it, the small 13% spam minority might end up completely missing from the test set by chance. Stratification forces both splits to keep the same 13/87 ratio.

**3. The pipeline (TF-IDF + Naive Bayes):**

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",     # remove common words: "the", "is", "a", "and"
        max_features=5000,        # keep only top 5000 most important words
        ngram_range=(1, 2),       # use single words AND pairs of words
        min_df=2,                 # word must appear in at least 2 documents
        sublinear_tf=True,        # use log(tf+1) instead of raw count
    )),
    ("model", MultinomialNB(alpha=0.1)),  # Laplace smoothing
])
```

Why these settings:
- **`stop_words="english"`** — words like "the", "is", "at" appear everywhere so they carry no signal. Removing them reduces noise and feature count.
- **`ngram_range=(1, 2)`** — single words catch "FREE" and "URGENT", but word pairs catch "free entry", "account compromised". Bigrams roughly double the feature count but improve accuracy for phrases.
- **`max_features=5000`** — the full vocabulary might be 10,000+ unique words. Limiting to 5000 keeps the model fast while capturing the most informative terms.
- **`min_df=2`** — a word that appears in only 1 message is probably noise (a typo or name). Requiring 2 appearances filters those out.
- **`sublinear_tf=True`** — raw word counts are dominated by very frequent words. Log scaling smooths the difference between "appears 5 times" and "appears 500 times."
- **`alpha=0.1`** — Laplace smoothing prevents zero probabilities when a test message contains a word the model never saw during training. Without smoothing, one unseen word would make the entire probability zero.

**4. Cross-validation:**

```python
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
```

This splits the training data into 5 parts, trains on 4 and tests on 1, then rotates through all 5 combinations. It tells you how stable the model is — if the 5 scores are close together (e.g., 0.905 to 0.940), the model is reliable. If they swing wildly (e.g., 0.5 to 0.95), the model is overfitting or the data is messy.

**5. Evaluation:**

```python
y_pred = pipeline.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)
```

The confusion matrix for this model looks like this (1,115 test messages):

```
                 Predicted
                 Ham   Spam
  Actual Ham  [ 963     3]    ← correctly identified 963 ham, misclassified 3 ham as spam
         Spam [ 17   132]    ← missed 17 spam, correctly caught 132 spam
```

- **97.78% precision** — when the model says "spam", it is correct 97.78% of the time. Only 3 legitimate messages out of 966 would be flagged.
- **88.59% recall** — out of 149 real spam messages, the model caught 132 and missed 17.
- The 17 missed spam messages are probably cleverly worded or use uncommon vocabulary.

**6. Saving the model:**

```python
joblib.dump(pipeline, os.path.join(MODEL_DIR, "spam_model.pkl"))
```

`joblib` is used instead of `pickle` because it is much more efficient for objects that contain large numpy arrays (like our TF-IDF sparse matrix). The entire pipeline — both the vectorizer and the classifier — is saved as one object. When the API loads it, calling `pipeline.predict()` automatically runs the text through TF-IDF then Naive Bayes in one step.

**7. Saving metrics:**

```python
metrics = {
    "accuracy":  round(float(acc),  6),
    "precision": round(float(prec), 6),
    "recall":    round(float(rec),  6),
    "f1":        round(float(f1),  6),
    "cv_f1_mean": round(float(cv_scores.mean()), 6),
    "cv_f1_std":  round(float(cv_scores.std()),  6),
    "confusion_matrix": cm.tolist(),
    "n_samples": int(len(texts)),
    "n_spam": int(n_spam),
    "n_ham": int(n_ham),
    "n_features": int(pipeline.named_steps["tfidf"].max_features or 0),
}
```

These metrics are written to `model/metrics.json` as JSON so the API can read them without needing scikit-learn installed on the server side.

---

### `api/main.py` — FastAPI Server

This file is the bridge between the trained model and the web. When it starts, it loads the saved pickle file and the metrics JSON, then exposes them through HTTP endpoints.

**Loading the model at startup:**

```python
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "model" / "spam_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

model   = joblib.load(str(MODEL_PATH))
with open(METRICS_PATH) as f:
    metrics = json.load(f)
```

The model is loaded once when the server starts, not on every request. This is the standard pattern — load heavy objects at startup, then serve predictions from memory.

**CORS middleware:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This allows any origin to call the API. Without it, browsers would block cross-origin requests from the frontend. In production, you would replace `"*"` with your actual frontend domain.

**Endpoint 1 — `GET /health`:**

```python
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "running",
        "model": "TF-IDF + MultinomialNB",
        "features": metrics["n_features"],
        "accuracy": f"{metrics['accuracy']:.2%}",
        "n_samples": metrics["n_samples"],
    }
```

The `response_model` tells FastAPI to validate and format the output according to the `HealthResponse` schema. This endpoint is used by the frontend to display model stats on page load.

**Endpoint 2 — `GET /metrics`:**

```python
@app.get("/metrics")
def get_metrics():
    return metrics
```

Simply returns the full metrics dictionary loaded from `metrics.json`. Useful for monitoring or dashboards.

**Endpoint 3 — `POST /predict`:**

```python
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    cleaned = clean_text(req.message)           # apply same cleaning used during training
    probs = model.predict_proba([cleaned])[0]    # [P(ham), P(spam)]
    pred  = model.predict([cleaned])[0]
    label = "SPAM" if pred == 1 else "HAM"
    confidence = float(max(probs))
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "spam_probability": round(float(probs[1]), 4),
        "ham_probability":  round(float(probs[0]), 4),
    }
```

How it works:
- `clean_text(req.message)` applies the same cleaning used during training (lowercase, strip URLs, phones, emails, punctuation).
- `model.predict_proba([cleaned])` returns `[P(ham), P(spam)]` — two probabilities that sum to 1.0. For example `[0.04, 0.96]` means 4% ham, 96% spam.
- `model.predict([cleaned])` returns the class label (0 or 1) based on the higher probability.
- `confidence` is the maximum probability — how sure the model is.
- Both probabilities are rounded to 4 decimal places for clean JSON output.

**Serving the frontend:**

```python
@app.get("/", include_in_schema=False)
async def serve_root(request: Request):
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend(request: Request, path: str):
    file_path = FRONTEND_DIR / path
    if path and file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(FRONTEND_DIR / "index.html")
```

The first handler serves `index.html` at the root URL `/`. The second is a catch-all that serves any file from the `frontend/` folder, falling back to `index.html` if the file does not exist. This lets you navigate directly to sub-pages if you ever split the frontend into multiple files.

---

### `frontend/index.html` — Web UI

This is a single-file frontend with no build step. All CSS is in a `<style>` block and all logic is in a `<script>` block.

**HTML structure:**

The page has four main sections:
1. **Stats bar** — three boxes showing Accuracy, Messages Trained, and Features (loaded from the API)
2. **Textarea + button** — where the user types or pastes a message
3. **Result display** — hidden by default, shows SPAM/HAM badge, confidence, and a dual probability bar
4. **Example chips** — six pre-written messages the user can click to instantly test

**CSS design choices:**

- Dark theme using Tailwind-inspired color palette (`#0f172a` background, `#1e293b` card, `#6366f1` indigo accent)
- The result box changes background color based on result: red tint for spam, green for ham
- The probability bar is a flex container with two colored spans that animate their width over 0.5 seconds
- All interactive elements (chips, button, textarea focus) have hover/focus transitions

**JavaScript logic:**

```javascript
const API = "/predict";
```

The API path is relative (`/predict`) because the frontend is served from the same origin as the backend.

**`loadStats()` — runs on page load:**

```javascript
(async function loadStats() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    document.getElementById("sAccuracy").textContent = d.accuracy;
    document.getElementById("sSamples").textContent  = d.n_samples.toLocaleString();
    document.getElementById("sFeatures").textContent = d.features.toLocaleString();
  } catch(e) { /* silently ignore */ }
})();
```

This is an IIFE (immediately invoked function expression) that fetches from `/health` and populates the three stat boxes. The `try/catch` ensures the page does not break if the API is not running.

**`predict()` — the main function:**

```javascript
async function predict() {
  const text = document.getElementById("msg").value.trim();
  if (!text) return;

  const btn = document.getElementById("btn");
  const res = document.getElementById("result");

  btn.disabled = true;
  btn.textContent = "Analysing...";

  try {
    const r = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const d = await r.json();

    // Update the result box
    res.className = "result " + d.label.toLowerCase();
    document.getElementById("badge").textContent = d.label === "SPAM" ? "⚠ SPAM" : "✅ HAM";
    document.getElementById("meta").textContent = `Confidence: ${(d.confidence * 100).toFixed(1)}%`;

    // Animate the probability bar
    document.getElementById("spamBar").style.width = (d.spam_probability * 100).toFixed(0) + "%";
    document.getElementById("hamBar").style.width  = (d.ham_probability * 100).toFixed(0) + "%";

    res.style.display = "block";

  } catch (e) {
    res.className = "result spam";
    document.getElementById("badge").textContent = "⚠ ERROR";
    document.getElementById("meta").textContent = "Is the API running? Start with: python -m api.main";
    res.style.display = "block";
  }

  btn.disabled = false;
  btn.textContent = "Check Now";
}
```

The flow:
1. Read the textarea value, trim whitespace, return early if empty
2. Disable the button and change text to "Analysing..."
3. POST to `/predict` with JSON body `{"message": "..."}`
4. On success: parse the JSON response, color the result box, update the badge and confidence text, animate the probability bars
5. On failure: show an error message reminding the user to start the API server
6. Re-enable the button

---

## Requirements

- Python 3.8 or newer
- scikit-learn
- pandas
- FastAPI
- uvicorn
- joblib
- numpy

## Install

### 1. Install the Python packages

```powershell
pip install -r requirements.txt
```

### 2. Dataset

The SMS Spam Collection (5,572 messages) is downloaded automatically during training. No manual setup needed. It is saved to `data/spam.csv` in TSV format.

## Run

### Step 1: Train the model

Open a terminal in the project folder and run:

```powershell
python -m src.train
```

This trains the model, prints cross-validation scores and test performance, and saves the artifacts to `model/`.

Sample training output:

```text
================================================== SPAM CLASSIFIER — Training Pipeline
==================================================

📂 Loading data from data/spam.csv...
   Delimiter: TAB
   Header: False | First field: 'ham'
📊 Dataset: 5572 messages (747 spam, 4825 ham)
   Spam ratio: 13.4%
📚 Training set: 4457 | 🧪 Test set: 1115

🔧 Building pipeline: TF-IDF → MultinomialNB
🏋️ Training...
✅ Done

🔄 Running 5-fold cross-validation...
   CV F1 scores: ['0.932', '0.918', '0.940', '0.905', '0.934']
   Mean F1: 0.926 (±0.013)

================================================== 📈 TEST SET PERFORMANCE
==================================================
   Accuracy : 98.21%
   Precision: 97.78% (flagged spam → actually spam)
   Recall   : 88.59% (caught spam out of all real spam)
   F1 Score : 92.96% (balanced measure)

   Confusion Matrix:
                Predicted
                Ham   Spam
   Actual Ham  [ 963     3]
          Spam [ 17   132]
==================================================

💾 Saved: model/spam_model.pkl
💾 Saved: model/metrics.json
🎉 Training complete! Model ready for serving.
```

### Step 2: Start the API server

In a second terminal:

```powershell
python -m api.main
```

Terminal output:

```text
 Loading model...
 Model loaded — 5000 features, 98.2% accuracy

 Server running on http://127.0.0.1:8000
   UI    → http://127.0.0.1:8000/
   Docs  → http://127.0.0.1:8000/docs
```

### Step 3: Open in your browser

Navigate to:

- **[http://localhost:8000](http://localhost:8000)** — Web UI
- **[http://localhost:8000/docs](http://localhost:8000/docs)** — Swagger API docs

---

## Output

After training, the `model/` folder contains:

```
model/
├── spam_model.pkl   ← Trained TF-IDF + MultinomialNB pipeline
└── metrics.json     ← Accuracy, precision, recall, F1, confusion matrix
```

The API returns predictions in this format:

```json
{
  "label": "SPAM",
  "confidence": 0.9597,
  "spam_probability": 0.9597,
  "ham_probability": 0.0403
}
```

The health check at `GET /health`:

```json
{
  "status": "running",
  "model": "TF-IDF + MultinomialNB",
  "features": 5000,
  "accuracy": "98.21%",
  "n_samples": 5572
}
```

---

## Notes

- The dataset has a natural class imbalance (~13% spam, ~87% ham) which is realistic for real SMS traffic. Stratified splitting ensures the model sees both classes during training and testing.
- The model is trained once and served statically — to retrain with new data, just re-run `python -m src.train` and restart the server.
- The frontend makes relative API calls (`/predict`, `/health`), so it must be served from the same origin as the backend (which the FastAPI server handles automatically).
- For production use, consider adding a larger dataset, model versioning, rate limiting, and authentication on the API endpoints.
- The old `train.py` and `main.py` from the initial single-file version are still in the project root for reference.

## Project Fit

This project follows the same lightweight, self-contained approach as the other tools in the workspace:

- small Python scripts as the main entry point
- dependencies listed in `requirements.txt`
- output (model artifacts) saved to a project-specific `model/` folder
- single HTML file for the frontend with no build step
- README written for GitHub-ready documentation

## Author

Created and maintained by `daxler_boi`.
