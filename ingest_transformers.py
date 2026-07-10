"""Ingestion method 1: sentence-transformer embeddings in a Chroma vector DB.

Embeds each comment with the BAAI/bge-small-en-v1.5 model and stores the
vectors, documents, and labels in a persistent Chroma collection. Later a
classifier can query the collection (e.g. nearest-neighbour vote) to decide
whether a new comment is toxic.

Usage:
    python ingest_transformers.py training_dataset.csv
    python ingest_transformers.py eval_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from dataset import ROOT, load_dataset

# Use the model already downloaded under models/bge-small-en-v1.5 so ingestion
# runs fully offline (no Hugging Face Hub download at runtime).
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_PATH = ROOT / "models" / "bge-small-en-v1.5"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "toxic_comments"


def ingest(dataset_path: Path, chroma_dir: Path, reset: bool) -> None:
    data = load_dataset(dataset_path)
    print(f"Loaded {len(data)} rows from {dataset_path.name}")

    client = chromadb.PersistentClient(path=str(chroma_dir))

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model not found at {MODEL_PATH}. "
            f"Download {MODEL_NAME} into that directory first."
        )
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=str(MODEL_PATH)
    )

    if reset:
        # Drop any prior collection so re-runs don't duplicate documents.
        try:
            client.delete_collection(COLLECTION_NAME)
        except (ValueError, chromadb.errors.NotFoundError):
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"model": MODEL_NAME, "hnsw:space": "cosine"},
    )

    # Stable ids keyed by source + row index make re-ingestion idempotent
    # (an upsert overwrites the same id rather than appending a duplicate).
    stem = dataset_path.stem
    ids = [f"{stem}-{i}" for i in range(len(data))]
    metadatas = [{"label": label, "source": stem} for label in data.labels]

    collection.upsert(ids=ids, documents=data.texts, metadatas=metadatas)

    print(
        f"Ingested {len(data)} documents into collection "
        f"'{COLLECTION_NAME}' at {chroma_dir} (now {collection.count()} total)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        default=None,
        help="CSV file to ingest (default: training_dataset.csv).",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=CHROMA_DIR,
        help="Directory for the persistent Chroma database.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Upsert into the existing collection instead of recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest(args.dataset, args.chroma_dir, reset=not args.no_reset)


if __name__ == "__main__":
    main()
