"""Download the BGE model into models/bge-small-en-v1.5/.

Used by CI (and handy for a fresh local checkout) so the model the code expects
at that path is present without committing 130 MB to git. Skips the onnx/ export
we don't use.

    python scripts/download_model.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download

TARGET = Path(__file__).resolve().parent.parent / "models" / "bge-small-en-v1.5"


def main() -> None:
    snapshot_download(
        repo_id="BAAI/bge-small-en-v1.5",
        local_dir=str(TARGET),
        ignore_patterns=["onnx/*", "*.onnx"],
    )
    print(f"Downloaded BGE model to {TARGET}")


if __name__ == "__main__":
    main()
