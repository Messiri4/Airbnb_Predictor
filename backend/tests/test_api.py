"""API endpoint tests: real saved model for integration, mock model for isolation."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.inference import InferenceError, PricePredictor
from app.main import app, predictor
from tests.conftest import MODEL_PATH, VALID_LISTING


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:  # runs lifespan (loads real model)
        yield c


class TestHealth:
    def test_health_reports_loaded_model(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_name"] == "Tuned Log-XGBoost"
        assert body["model_version"] == "v1"
        assert body["sklearn_version"]


class TestPredictEndpoint:
    def test_valid_prediction(self, client: TestClient) -> None:
        r = client.post("/api/predict", json=VALID_LISTING)
        assert r.status_code == 200
        body = r.json()
        assert body["currency"] == "USD"
        assert isinstance(body["predicted_price"], float)
        assert math.isfinite(body["predicted_price"])
        assert body["model_name"] == "Tuned Log-XGBoost"

    def test_prediction_matches_saved_model_directly(self, client: TestClient) -> None:
        # Integration consistency: API result == calling the saved artifact directly.
        direct = PricePredictor(MODEL_PATH)
        direct.load()
        expected = round(direct.predict(VALID_LISTING), 2)
        r = client.post("/api/predict", json=VALID_LISTING)
        assert r.json()["predicted_price"] == pytest.approx(expected)

    def test_invalid_category_rejected(self, client: TestClient) -> None:
        r = client.post("/api/predict", json={**VALID_LISTING, "neighbourhood_group": "Paris"})
        assert r.status_code == 422

    def test_missing_field_rejected(self, client: TestClient) -> None:
        broken = {k: v for k, v in VALID_LISTING.items() if k != "room_type"}
        r = client.post("/api/predict", json=broken)
        assert r.status_code == 422

    def test_out_of_range_values_rejected(self, client: TestClient) -> None:
        r = client.post("/api/predict", json={**VALID_LISTING, "latitude": 999})
        assert r.status_code == 422
        r = client.post("/api/predict", json={**VALID_LISTING, "minimum_nights": 0})
        assert r.status_code == 422

    def test_no_stack_trace_in_error(self, client: TestClient) -> None:
        r = client.post("/api/predict", json={**VALID_LISTING, "room_type": "Igloo"})
        assert "Traceback" not in r.text


class TestMockModelIsolation:
    def test_unloaded_model_returns_503(self) -> None:
        with (
            patch.object(predictor, "_pipeline", None),
            patch.object(PricePredictor, "is_loaded", property(lambda self: False)),
            TestClient(app) as c,
        ):
            r = c.post("/api/predict", json=VALID_LISTING)
            assert r.status_code == 503
            assert "not available" in r.json()["detail"].lower()

    def test_model_failure_maps_to_422(self, client: TestClient) -> None:
        with patch.object(predictor, "predict", side_effect=InferenceError("boom")):
            r = client.post("/api/predict", json=VALID_LISTING)
            assert r.status_code == 422
