# Serving image for the BGE + Logistic Regression pipeline.
#
# It bakes in the PRE-TRAINED artifacts (the local BGE model + models_out/logreg.joblib)
# and just serves them -- no ingestion or training happens during the build.
FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch. The droplet has no GPU, and the default torch wheel drags in
# ~1GB of CUDA libraries we don't need, so pull the much smaller CPU build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Python runtime deps (fastapi, uvicorn, sentence-transformers, sklearn, joblib).
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code.
COPY predict.py serve.py ./

# Pre-trained artifacts, baked in. These are gitignored, so they must exist
# locally before building (run the classify step once) or be rsynced to the
# droplet -- see DEPLOY.md.
COPY models/bge-small-en-v1.5/ ./models/bge-small-en-v1.5/
COPY models_out/logreg.joblib ./models_out/logreg.joblib

# Never reach out to the Hugging Face Hub at runtime; the model is local.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The git commit this image was built from, passed by CI (--build-arg GIT_SHA=...)
# and exposed by the app at /version. Declared late so a new SHA only rebuilds
# this tiny layer, not the expensive torch/model layers above.
ARG GIT_SHA=unknown
ENV GIT_SHA=${GIT_SHA}

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
