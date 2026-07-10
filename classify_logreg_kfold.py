"""K-fold cross-validation for the BGE + Logistic Regression pipeline.

This is the *evaluation* counterpart to classify_logreg.py. Where that script
does a single naive 80/20 train/eval split, this one pools ALL the embeddings
and runs stratified k-fold cross-validation.

Why: on a small dataset a single split is a high-variance estimate -- the score
depends heavily on which rows happened to land in eval. K-fold rotates the
held-out fold so every comment is used for both training and evaluation (never
at once), and the mean +/- std across folds tells you how noisy the estimate is.
A difference between models only matters if it clears that std.

This does NOT save a model -- it measures how well the pipeline generalizes. The
deployed model still comes from classify_logreg.py.

Run ingestion first so the embeddings exist in Chroma (both splits -- they get
pooled back together here):
    python ingest_transformers.py training_dataset.csv
    python ingest_transformers.py eval_dataset.csv

Then:
    python classify_logreg_kfold.py
"""

import chromadb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "toxic_comments"
N_SPLITS = 5


def load_all(collection):
    """Load every embedding + label from the collection (all sources pooled)."""
    result = collection.get(include=["embeddings", "metadatas"])
    X = np.array(result["embeddings"])
    y = np.array([meta["label"] for meta in result["metadatas"]])
    return X, y


def main():
    # Read the embeddings the ingestion step stored in Chroma.
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    X, y = load_all(collection)
    classes, counts = np.unique(y, return_counts=True)
    print(f"Loaded {len(X)} embeddings. Class balance: {dict(zip(classes, counts))}")

    # Stratified so every fold keeps the toxic / non-toxic ratio; shuffle so the
    # folds don't just mirror the order rows were ingested in.
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000)

    # macro-F1 alongside accuracy: it weights both classes equally, which is the
    # more honest summary on a small, roughly balanced dataset.
    scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
    acc = scores["test_accuracy"]
    f1 = scores["test_f1_macro"]

    print(f"\nStratified {N_SPLITS}-fold cross-validation (BGE + Logistic Regression)\n")
    print("fold   accuracy   f1_macro")
    for i, (a, f) in enumerate(zip(acc, f1), start=1):
        print(f"  {i}       {a:.3f}      {f:.3f}")

    print(f"\naccuracy: {acc.mean():.3f} +/- {acc.std():.3f}")
    print(f"f1_macro: {f1.mean():.3f} +/- {f1.std():.3f}")
    print("\n(Treat a difference between models as real only if it clears the +/- std.)")


if __name__ == "__main__":
    main()
