"""Pipeline 2 classifier: LightGBM on TF-IDF features.

Loads the TF-IDF feature matrices (created by ingest_tfidf.py), trains a
LightGBM classifier on the training split, and evaluates it on the eval split.

Run the ingestion first:
    python ingest_tfidf.py
    python ingest_tfidf.py --dataset eval_dataset.csv --transform-only

Then:
    python classify_lightgbm.py
"""

from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report

ARTIFACTS_DIR = Path("tfidf_artifacts")
OUTPUT_PATH = Path("models_out/lightgbm.joblib")


def load_split(split):
    """Load the TF-IDF feature matrix and labels for one split."""
    X = load_npz(ARTIFACTS_DIR / f"{split}_features.npz")
    y = joblib.load(ARTIFACTS_DIR / f"{split}_labels.joblib")
    return X, y


def main():
    # Load the TF-IDF features produced during ingestion.
    X_train, y_train = load_split("training_dataset")
    X_eval, y_eval = load_split("eval_dataset")
    print(f"Training on {X_train.shape[0]} rows, evaluating on {X_eval.shape[0]}.")

    # Train LightGBM. This dataset is tiny (79 rows) and the TF-IDF matrix is
    # very sparse, so LightGBM's defaults (min_data_in_bin=3,
    # min_child_samples=20) are too strict and it builds no splits at all.
    # Lowering them lets it actually learn from the features.
    model = LGBMClassifier(
        random_state=42,
        min_data_in_bin=1,
        min_child_samples=5,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate on the eval split.
    predictions = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, predictions)
    print(f"\nEval accuracy: {accuracy:.3f}\n")
    print(classification_report(y_eval, predictions, zero_division=0))

    # Save the trained model.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
