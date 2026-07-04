"""Smoke tests for the FastAPI app.

The ML layer is mocked out, so these run in milliseconds and need no model,
torch, or trained artifacts -- they check the API wiring (routes, response
shape, validation), not the classifier's accuracy.
"""

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

import serve


@dataclass
class FakePrediction:
    label: str
    confidence: float
    probabilities: dict


@pytest.fixture
def client(monkeypatch):
    # Replace the real (model-loading) prediction calls with fast fakes.
    monkeypatch.setattr(
        serve, "predict_bge",
        lambda text: FakePrediction("toxic", 0.9, {"toxic": 0.9, "non_toxic": 0.1}),
    )
    monkeypatch.setattr(
        serve, "predict_tfidf",
        lambda text: FakePrediction("non_toxic", 0.6, {"toxic": 0.4, "non_toxic": 0.6}),
    )
    # The startup hook would otherwise load the BGE model; no-op it for tests.
    monkeypatch.setattr(serve, "warmup_bge", lambda: None)
    with TestClient(serve.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_is_it_toxic_returns_both_pipelines(client):
    r = client.post("/is-it-toxic", json={"text": "you are the worst"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "you are the worst"

    bge = body["bge_logreg"]
    assert bge["label"] == "toxic"
    assert 0.0 <= bge["confidence"] <= 1.0
    assert set(bge["probabilities"]) == {"toxic", "non_toxic"}

    # TF-IDF pipeline is present because its (mocked) prediction succeeded.
    assert body["tfidf_lightgbm"]["label"] == "non_toxic"


def test_missing_text_is_rejected(client):
    r = client.post("/is-it-toxic", json={})
    assert r.status_code == 422  # pydantic validation error


def test_version_reports_git_sha(client, monkeypatch):
    monkeypatch.setattr(serve, "GIT_SHA", "abc123")
    r = client.get("/version")
    assert r.status_code == 200
    assert r.json() == {"git_sha": "abc123"}
