"""API smoke tests (no model required for /health)."""

from fastapi.testclient import TestClient

from app.api import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict_without_model_returns_503_or_ok():
    client = TestClient(app)
    # If model missing → 503; if present → 200/400 depending on payload.
    r = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code in (400, 503)
