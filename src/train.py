"""
Train a spam classifier using TF-IDF + MultinomialNB.
Full pipeline: load data -> train -> evaluate -> save model + metrics.
"""
import sys, io, json, os
from pathlib import Path

# Ensure src/ is on the path so we can import preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import load_csv, clean_text

# ============================================================================
# Config
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "spam.csv"
MODEL_DIR  = BASE_DIR / "model"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# ============================================================================

# Load data
print("=" * 50)
print("  SPAM CLASSIFIER - Training Pipeline")
print("=" * 50)

print(f"\nLoading data from {DATA_PATH}...")
texts, labels = load_csv(DATA_PATH)
labels = np.array(labels)

n_spam = labels.sum()
n_ham = len(labels) - n_spam
print(f"Dataset: {len(texts)} messages  ({n_spam} spam, {n_ham} ham)")
print(f"   Spam ratio: {n_spam / len(labels):.1%}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
)
print(f"\nTraining set: {len(X_train)}  |  Test set: {len(X_test)}")

# Build pipeline
print(f"\nBuilding pipeline: TF-IDF -> MultinomialNB")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),       # single words + word pairs
        min_df=2,                 # word must appear in at least 2 docs
        sublinear_tf=True,        # use log scaling for term frequency
    )),
    ("model", MultinomialNB(alpha=0.1)),     # small smoothing
])

# Train
print("Training...")
pipeline.fit(X_train, y_train)
print("Done")

# Cross-validation on training set
print(f"\nRunning {CV_FOLDS}-fold cross-validation...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=CV_FOLDS, scoring="f1")
print(f"   CV F1 scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"   Mean F1: {cv_scores.mean():.3f}  (+/-{cv_scores.std():.3f})")

# Evaluate on test set
print(f"\n{'=' * 50}")
print("  TEST SET PERFORMANCE")
print(f"{'=' * 50}")

y_pred = pipeline.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print(f"   Accuracy :  {acc:.2%}")
print(f"   Precision:  {prec:.2%}  (flagged spam -> actually spam)")
print(f"   Recall   :  {rec:.2%}  (caught spam out of all real spam)")
print(f"   F1 Score :  {f1:.2%}  (balanced measure)")
print(f"\n   Confusion Matrix:")
print(f"                Predicted")
print(f"                Ham   Spam")
print(f"   Actual Ham  [{cm[0][0]:4d}  {cm[0][1]:4d}]")
print(f"          Spam [{cm[1][0]:4d}  {cm[1][1]:4d}]")
print(f"{'=' * 50}")

# Manual test with sample messages
print(f"\nSAMPLE PREDICTIONS")
print(f"{'-' * 50}")

samples = [
    "Win a free iPhone now click here to claim",
    "Hey can you send me the project files",
    "URGENT: Your account has been compromised act now",
    "Thanks for your help with the presentation",
    "Exclusive deal 90% off limited time only",
    "Meeting at 3pm in the conference room",
]
probs = pipeline.predict_proba(samples)
preds = pipeline.predict(samples)

for msg, pred, prob in zip(samples, preds, probs):
    label = "[SPAM]" if pred == 1 else "[HAM]"
    conf = prob[1] if pred == 1 else prob[0]
    print(f"   {label}  ({conf:.0%})  - {msg[:50]}")

# Save model + metrics
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\nSaving model...")
joblib.dump(pipeline, os.path.join(MODEL_DIR, "spam_model.pkl"))
print(f"   - {MODEL_DIR}/spam_model.pkl")

# Save metrics
n_features = pipeline.named_steps["tfidf"].max_features or "auto"
metrics = {
    "accuracy":  round(float(acc),  6),
    "precision": round(float(prec), 6),
    "recall":    round(float(rec),  6),
    "f1":        round(float(f1),  6),
    "cv_f1_mean":    round(float(cv_scores.mean()), 6),
    "cv_f1_std":     round(float(cv_scores.std()),  6),
    "confusion_matrix": cm.tolist(),
    "n_samples":     int(len(texts)),
    "n_spam":        int(n_spam),
    "n_ham":         int(n_ham),
    "n_train":       int(len(X_train)),
    "n_test":        int(len(X_test)),
    "n_features":    int(pipeline.named_steps["tfidf"].max_features or 0),
    "vectorizer":    "TF-IDF (1-2 grams)",
    "model":         "MultinomialNB (alpha=0.1)",
    "random_state":  RANDOM_STATE,
}

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"   - {MODEL_DIR}/metrics.json")

print(f"\nTraining complete! Model ready for serving.")
print(f"   Start API: python -m api.main")