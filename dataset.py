"""Shared dataset loading utilities.

Both ingestion methods read the same CSV format produced for this project:

    label,text
    toxic,"Some comment"
    non_toxic,"Another comment"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Project paths. Datasets live alongside this module.
ROOT = Path(__file__).resolve().parent
TRAINING_CSV = ROOT / "training_dataset.csv"
EVAL_CSV = ROOT / "eval_dataset.csv"

# The two labels present in the dataset.
LABELS = ("toxic", "non_toxic")


@dataclass
class Dataset:
    """A loaded dataset: parallel lists of texts and their labels."""

    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)


def load_dataset(path: str | Path) -> Dataset:
    """Load a `label,text` CSV into a :class:`Dataset`.

    Rows with a missing/blank text or an unrecognised label are dropped so the
    downstream ingestion steps only ever see clean records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = {"label", "text"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(sorted(missing))}"
        )

    # Normalise and drop unusable rows.
    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].astype(bool)]

    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(
            f"{path} contains unexpected label(s): {', '.join(sorted(unknown))}. "
            f"Expected one of {LABELS}."
        )

    return Dataset(texts=df["text"].tolist(), labels=df["label"].tolist())
