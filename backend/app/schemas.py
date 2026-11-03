"""Pydantic request/response schemas matching the saved model's exact inputs."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

NEIGHBOURHOOD_GROUPS = ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
ROOM_TYPES = ("Entire home/apt", "Private room", "Shared room")


class ListingFeatures(BaseModel):
    """Raw listing features — identical names/order to the saved pipeline's inputs."""

    neighbourhood_group: str = Field(..., description="NYC borough name")
    room_type: str = Field(..., description="Airbnb room type")
    latitude: float = Field(..., ge=-90, le=90, description="Listing latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Listing longitude")
    minimum_nights: int = Field(..., ge=1, le=1125, description="Minimum nights to book")
    number_of_reviews: int = Field(..., ge=0, description="Total reviews")
    reviews_per_month: float = Field(..., ge=0, le=60, description="Reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Listings by this host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available per year")

    @field_validator("neighbourhood_group")
    def validate_group(cls, v: str) -> str:
        v = v.strip()
        if v not in NEIGHBOURHOOD_GROUPS:
            raise ValueError(f"neighbourhood_group must be one of {NEIGHBOURHOOD_GROUPS}")
        return v

    @field_validator("room_type")
    def validate_room(cls, v: str) -> str:
        v = v.strip()
        if v not in ROOM_TYPES:
            raise ValueError(f"room_type must be one of {ROOM_TYPES}")
        return v


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated nightly price in USD")
    currency: str = "USD"
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    sklearn_version: str | None = None
    xgboost_version: str | None = None
