"""Shared prediction helpers for both pipelines.

Each pipeline is **lazy-loaded** on first use and then cached, so importing this
module does not require every artifact to be present. In particular the
deployment image only ships the BGE + Logistic Regression artifacts, so
`predict_bge` works there while `predict_tfidf` would only load its artifacts if
called.

    predict_bge(text)     -> Prediction   (BGE embeddings + Logistic Regression)
    predict_tfidf(text)   -> Prediction   (TF-IDF + LightGBM)

Both the FastAPI server (serve.py) and the exploration notebook use these so
there is exactly one place that knows how a raw comment turns into a label.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
BGE_MODEL_PATH = ROOT / "models" / "bge-small-en-v1.5"
LOGREG_PATH = ROOT / "models_out" / "logreg.joblib"
VECTORIZER_PATH = ROOT / "tfidf_artifacts" / "vectorizer.joblib"
LIGHTGBM_PATH = ROOT / "models_out" / "lightgbm.joblib"


@dataclass
class Prediction:
    label: str  # "toxic" or "non_toxic"
    confidence: float  # probability of the predicted label, 0..1
    probabilities: dict  # {label: probability} for every class


# --- Lazy loaders (each runs once, then is cached) -----------------------


@lru_cache(maxsize=1)
def _load_bge():
    """BGE pipeline: the sentence-transformer embedder + Logistic Regression head."""
    embedder = SentenceTransformer(str(BGE_MODEL_PATH))
    logreg = joblib.load(LOGREG_PATH)
    return embedder, logreg


@lru_cache(maxsize=1)
def _load_tfidf():
    """TF-IDF pipeline: the fitted vectorizer + the LightGBM classifier."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    lightgbm = joblib.load(LIGHTGBM_PATH)
    return vectorizer, lightgbm


def _to_prediction(model, features) -> Prediction:
    """Turn a model's predict_proba output into a Prediction."""
    proba = model.predict_proba(features)[0]
    classes = list(model.classes_)
    probabilities = {label: float(p) for label, p in zip(classes, proba)}
    best_label = max(probabilities, key=probabilities.get)
    return Prediction(
        label=best_label,
        confidence=probabilities[best_label],
        probabilities=probabilities,
    )


def warmup_bge() -> None:
    """Load the BGE pipeline ahead of time (e.g. at server startup)."""
    _load_bge()


def embed(texts):
    """Return BGE embeddings for a list of texts (used by the notebook plots)."""
    embedder, _ = _load_bge()
    return embedder.encode(list(texts))


def predict_bge(text: str) -> Prediction:
    """Pipeline 1: embed the text with BGE, classify with Logistic Regression."""
    embedder, logreg = _load_bge()
    embedding = embedder.encode([text])
    return _to_prediction(logreg, embedding)


def predict_tfidf(text: str) -> Prediction:
    """Pipeline 2: vectorize the text with TF-IDF, classify with LightGBM."""
    vectorizer, lightgbm = _load_tfidf()
    features = vectorizer.transform([text])
    return _to_prediction(lightgbm, features)
