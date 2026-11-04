"""FastAPI application around the saved tuned Log-XGBoost pipeline."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import sklearn
import xgboost
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.inference import InferenceError, PricePredictor
from app.logging_setup import configure_logging
from app.schemas import HealthResponse, ListingFeatures, PredictionResponse

configure_logging()
logger = logging.getLogger(__name__)

predictor = PricePredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the saved artifact exactly once.
    try:
        predictor.load()
        logger.info("Model loaded; serving predictions.")
    except InferenceError as exc:
        logger.error("Model failed to load: %s", exc)
        # App still boots so /health can report the failure state.
    yield
    # Shutdown: nothing to release (pickle holds no external connections).


app = FastAPI(
    title="Airbnb Pricing Intelligence API",
    description=(
        "REST API for the saved tuned Log-XGBoost model (NYC Airbnb 2019). "
        "Predicts an appropriate nightly listing price in USD."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allowed origins come from the ALLOWED_ORIGINS env var (comma-separated).
# Defaults cover local Docker/dev. In production, set ALLOWED_ORIGINS to the
# deployed frontend's HTTPS origin (e.g. https://airbnb-frontend.onrender.com).
import os

DEFAULT_ORIGINS = ["http://localhost:5173", "http://localhost:8080"]
ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
] or DEFAULT_ORIGINS
logger.info("CORS allowed origins: %s", ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    logger.warning("Inference error: %s", exc)
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + model status."""
    info = predictor.info()
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_name=info.name,
        model_version=info.version,
        sklearn_version=sklearn.__version__,
        xgboost_version=xgboost.__version__,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(listing: ListingFeatures) -> PredictionResponse:
    """Estimate the nightly price (USD) for a listing."""
    if not predictor.is_loaded:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not available.")
    try:
        price = predictor.predict(listing.model_dump())
    except InferenceError as exc:
        logger.warning("Prediction failed validation/inference: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    info = predictor.info()
    return PredictionResponse(
        predicted_price=round(price, 2),
        currency="USD",
        model_name=info.name,
        model_version=info.version,
    )
