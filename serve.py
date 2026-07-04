"""FastAPI server exposing the /is-it-toxic endpoint.

The primary pipeline is BGE + Logistic Regression. If the TF-IDF + LightGBM
artifacts are also present (they are locally, but not in the deployment image),
the response includes that pipeline too so you can compare them.

Start it with:
    uvicorn serve:app --reload

Then try it:
    curl -X POST http://127.0.0.1:8000/is-it-toxic \
         -H "Content-Type: application/json" \
         -d '{"text": "You are the reason this team is failing."}'

Or open the interactive docs at http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_bge, predict_tfidf, warmup_bge

# The git commit this image was built from. CI bakes it in as a build arg ->
# env var (see Dockerfile); falls back to "unknown" for local runs.
GIT_SHA = os.getenv("GIT_SHA", "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the BGE model once at startup so the first request isn't slow.
    warmup_bge()
    yield


app = FastAPI(title="Is it toxic?", lifespan=lifespan)


class CommentIn(BaseModel):
    text: str


class PipelineResult(BaseModel):
    label: str
    confidence: float
    probabilities: dict


class ToxicResponse(BaseModel):
    text: str
    bge_logreg: PipelineResult
    # Only populated when the TF-IDF + LightGBM artifacts are available.
    tfidf_lightgbm: Optional[PipelineResult] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    """Report the exact git commit this image was built from."""
    return {"git_sha": GIT_SHA}


@app.post("/is-it-toxic", response_model=ToxicResponse)
def is_it_toxic(comment: CommentIn):
    bge = predict_bge(comment.text)
    response = ToxicResponse(
        text=comment.text,
        bge_logreg=PipelineResult(
            label=bge.label,
            confidence=bge.confidence,
            probabilities=bge.probabilities,
        ),
    )

    # Add the TF-IDF pipeline for comparison if its artifacts are present.
    try:
        tfidf = predict_tfidf(comment.text)
        response.tfidf_lightgbm = PipelineResult(
            label=tfidf.label,
            confidence=tfidf.confidence,
            probabilities=tfidf.probabilities,
        )
    except FileNotFoundError:
        pass  # deployment image ships BGE + LogReg only

    return response
