"""Shared pytest fixtures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "best_price_model.pkl")

VALID_LISTING: dict[str, Any] = {
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


@pytest.fixture()
def valid_listing() -> dict[str, Any]:
    return dict(VALID_LISTING)
