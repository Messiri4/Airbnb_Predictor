"""Model inference for the saved tuned Log-XGBoost pipeline.

The artifact is a complete sklearn Pipeline:
  ColumnTransformer(preprocessing) -> TransformedTargetRegressor(log1p/expm1) -> XGBRegressor

The inverse log transform lives INSIDE the artifact: `predict()` returns USD directly.
No manual expm1 is applied here.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# The exact feature order expected by the saved ColumnTransformer.
CATEGORICAL_FEATURES = ["neighbourhood_group", "room_type"]
NUMERICAL_FEATURES = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]
FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(Path(__file__).resolve().parents[2] / "models" / "best_price_model.pkl")
)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    artifact_path: str


class InferenceError(RuntimeError):
    """Raised when a prediction cannot be produced with the saved model."""


class PricePredictor:
    """Thin wrapper around the saved pipeline. Loads once, predicts many."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._pipeline: Any = None

    def load(self) -> None:
        """Load the saved artifact exactly as it was persisted (no fitting)."""
        path = Path(self.model_path)
        if not path.is_file():
            raise InferenceError(f"Model artifact not found at: {path}")
        logger.info("Loading model artifact from %s", path)
        with open(path, "rb") as fh:
            self._pipeline = pickle.load(fh)
        # Verify it is the expected structure before serving traffic.
        if not hasattr(self._pipeline, "predict") or not hasattr(self._pipeline, "steps"):
            raise InferenceError("Artifact is not a sklearn Pipeline; refusing to serve it.")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="Tuned Log-XGBoost",
            version="v1",
            artifact_path=str(self.model_path),
        )

    def predict(self, features: dict[str, Any]) -> float:
        """Predict nightly price in USD from a dict of raw feature values.

        Input dict must contain all FEATURE_ORDER keys. Categorical values are
        raw labels ("Brooklyn", "Private room"); the saved OneHotEncoder handles
        them. Output is already in USD thanks to the artifact's inverse transform.
        """
        if not self.is_loaded:
            raise InferenceError("Model is not loaded.")

        missing = [k for k in FEATURE_ORDER if k not in features]
        if missing:
            raise InferenceError(f"Missing required features: {missing}")

        frame = pd.DataFrame([{k: features[k] for k in FEATURE_ORDER}])
        try:
            prediction = self._pipeline.predict(frame)
        except Exception as exc:
            raise InferenceError(f"Model prediction failed: {exc}") from exc

        price = float(prediction[0])
        if not pd.notna(price):
            raise InferenceError("Model produced a non-finite prediction.")
        return price
