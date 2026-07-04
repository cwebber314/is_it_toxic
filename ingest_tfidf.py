"""Ingestion method 2: TF-IDF features with scikit-learn.

Fits a :class:`~sklearn.feature_extraction.text.TfidfVectorizer` on the
training comments and stores the fitted vectorizer, the transformed feature
matrix, and the labels. Downstream a classifier (e.g. LogisticRegression) can
load these artifacts to train/predict without re-fitting the vocabulary.

Important: the vectorizer must be *fit only on training data*. Eval data is
transformed with the already-fitted vectorizer so the vocabulary never leaks.

Usage:
    python ingest_tfidf.py                 # fit + transform the training set
    python ingest_tfidf.py --dataset eval_dataset.csv --transform-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import ROOT, TRAINING_CSV, load_dataset

ARTIFACTS_DIR = ROOT / "tfidf_artifacts"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"


def ingest(dataset_path: Path, artifacts_dir: Path, transform_only: bool) -> None:
    data = load_dataset(dataset_path)
    print(f"Loaded {len(data)} rows from {dataset_path.name}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = artifacts_dir / "vectorizer.joblib"

    if transform_only:
        # Reuse the vocabulary learned from the training set.
        if not vectorizer_path.exists():
            raise FileNotFoundError(
                f"No fitted vectorizer at {vectorizer_path}. "
                "Run without --transform-only on the training set first."
            )
        vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
        features = vectorizer.transform(data.texts)
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        features = vectorizer.fit_transform(data.texts)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Fitted vectorizer ({len(vectorizer.vocabulary_)} terms) -> {vectorizer_path}")

    # Persist the feature matrix and aligned labels for this split.
    stem = dataset_path.stem
    features_path = artifacts_dir / f"{stem}_features.npz"
    labels_path = artifacts_dir / f"{stem}_labels.joblib"
    save_npz(features_path, features)
    joblib.dump(data.labels, labels_path)

    print(
        f"Saved features {features.shape} -> {features_path.name} "
        f"and labels -> {labels_path.name} in {artifacts_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=TRAINING_CSV,
        help="CSV file to ingest (default: training_dataset.csv).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory for the fitted vectorizer and feature matrices.",
    )
    parser.add_argument(
        "--transform-only",
        action="store_true",
        help="Transform using an existing vectorizer instead of fitting a new one "
        "(use for the eval set).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(args.dataset, args.artifacts_dir, transform_only=args.transform_only)


if __name__ == "__main__":
    main()
