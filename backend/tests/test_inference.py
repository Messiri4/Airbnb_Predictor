"""Test the inference module against the real saved model artifact."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.inference import (
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERICAL_FEATURES,
    InferenceError,
    PricePredictor,
)

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "best_price_model.pkl")


@pytest.fixture(scope="module")
def predictor() -> PricePredictor:
    p = PricePredictor(MODEL_PATH)
    p.load()
    return p


VALID_LISTING = {
    "neighbourhood_group": "Brooklyn",
    "room_type": "Private room",
    "latitude": 40.68,
    "longitude": -73.95,
    "minimum_nights": 2,
    "number_of_reviews": 10,
    "reviews_per_month": 0.5,
    "calculated_host_listings_count": 1,
    "availability_365": 120,
}


class TestModelLoading:
    def test_loads_successfully(self, predictor: PricePredictor) -> None:
        assert predictor.is_loaded is True

    def test_missing_artifact_raises(self, tmp_path) -> None:
        p = PricePredictor(str(tmp_path / "nope.pkl"))
        with pytest.raises(InferenceError, match="not found"):
            p.load()

    def test_feature_order_has_nine_features(self) -> None:
        assert len(FEATURE_ORDER) == 9
        assert CATEGORICAL_FEATURES == ["neighbourhood_group", "room_type"]
        assert NUMERICAL_FEATURES[0] == "latitude"
        assert NUMERICAL_FEATURES[-1] == "availability_365"


class TestPredictions:
    def test_returns_finite_float(self, predictor: PricePredictor) -> None:
        price = predictor.predict(VALID_LISTING)
        assert isinstance(price, float)
        assert math.isfinite(price)

    def test_price_on_usd_scale(self, predictor: PricePredictor) -> None:
        # USD nightly prices in this dataset run ~$10 to ~$10,000; typical << $1000.
        price = predictor.predict(VALID_LISTING)
        assert 5 < price < 10000, f"Prediction {price} is outside the USD nightly scale"

    def test_prediction_is_deterministic(self, predictor: PricePredictor) -> None:
        assert predictor.predict(VALID_LISTING) == pytest.approx(
            predictor.predict(dict(VALID_LISTING))
        )

    def test_reference_value_matches_saved_model_behavior(self, predictor: PricePredictor) -> None:
        # Verified during the audit: this listing produced 74.93 USD from the saved artifact.
        price = predictor.predict(VALID_LISTING)
        assert price == pytest.approx(74.93, abs=0.05)

    def test_manhattan_entire_home_more_expensive_than_shared(self, predictor: PricePredictor) -> None:
        base = {**VALID_LISTING, "neighbourhood_group": "Manhattan", "room_type": "Entire home/apt"}
        shared = {**base, "room_type": "Shared room"}
        assert predictor.predict(base) > predictor.predict(shared)

    def test_no_training_occurs_at_inference(self, predictor: PricePredictor) -> None:
        # Calling predict repeatedly must not change the estimator (still unfitted-copy semantics).
        before = predictor.predict(VALID_LISTING)
        for _ in range(5):
            predictor.predict(VALID_LISTING)
        assert predictor.predict(VALID_LISTING) == pytest.approx(before)


class TestInvalidInputs:
    def test_missing_feature_raises(self, predictor: PricePredictor) -> None:
        broken = {k: v for k, v in VALID_LISTING.items() if k != "room_type"}
        with pytest.raises(InferenceError, match="Missing required features"):
            predictor.predict(broken)

    def test_unknown_category_still_predicts(self, predictor: PricePredictor) -> None:
        # The saved OneHotEncoder uses handle_unknown='ignore'; it must not crash.
        listing = {**VALID_LISTING, "neighbourhood_group": "Atlantis"}
        price = predictor.predict(listing)
        assert math.isfinite(price)
