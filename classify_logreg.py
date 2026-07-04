"""Pipeline 1 classifier: Logistic Regression on BGE embeddings.

Reads the sentence-transformer embeddings back out of the Chroma vector DB
(created by ingest_transformers.py), trains a Logistic Regression model on the
training split, and evaluates it on the eval split.

Run the ingestion first:
    python ingest_transformers.py
    python ingest_transformers.py --dataset eval_dataset.csv --no-reset

Then:
    python classify_logreg.py
"""

from pathlib import Path

import chromadb
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "toxic_comments"
OUTPUT_PATH = Path("models_out/logreg.joblib")


def load_split(collection, source):
    """Get the embeddings and labels for one split (training / eval) from Chroma."""
    result = collection.get(
        where={"source": source},
        include=["embeddings", "metadatas"],
    )
    embeddings = result["embeddings"]
    labels = [meta["label"] for meta in result["metadatas"]]
    return embeddings, labels


def main():
    # Open the Chroma collection built during ingestion.
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    # Pull the training and eval embeddings back out.
    X_train, y_train = load_split(collection, "training_dataset")
    X_eval, y_eval = load_split(collection, "eval_dataset")
    print(f"Training on {len(X_train)} embeddings, evaluating on {len(X_eval)}.")

    # Train Logistic Regression.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate on the eval split.
    predictions = model.predict(X_eval)
    accuracy = accuracy_score(y_eval, predictions)
    print(f"\nEval accuracy: {accuracy:.3f}\n")
    print(classification_report(y_eval, predictions))

    # Save the trained model.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)
    print(f"Saved model to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
