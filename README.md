# 🏡 Airbnb Pricing Intelligence — Decision Support System

A **Decision Support System (DSS)** that helps Airbnb hosts price their NYC listings.
The machine learning work is **complete** — this repository now serves the saved
**tuned Log-XGBoost** model through a production-style **FastAPI + React + Docker**
web application with automated **CI/CD**.

> The ML pipeline, model selection, tuning, and evaluation live in
> `Airbnb_Price_Prediction_updated.ipynb` and `PROJECT_CONTEXT.md`.
> **No model is trained or retuned at runtime or in CI — the saved artifact is used as-is.**

## 🧠 The Model

| | |
|---|---|
| Algorithm | XGBoost regressor, log-transformed target (`log1p` → `expm1`) |
| Tuning | RandomizedSearchCV (40 candidates × 5-fold CV, MAE-scored) |
| Artifact | `models/best_price_model.pkl` — a complete sklearn `Pipeline` |
| Includes | ColumnTransformer (OneHotEncoder + imputers) **and** the inverse log transform |
| Test MAE | **$58.19** (hold-out test set; see `PROJECT_CONTEXT.md`) |
| Predict output | USD nightly price, directly — no manual inverse transform |

**Inference inputs (exact schema):**

| Feature | Type | Values / Range |
|---|---|---|
| `neighbourhood_group` | string | Bronx, Brooklyn, Manhattan, Queens, Staten Island |
| `room_type` | string | Entire home/apt, Private room, Shared room |
| `latitude` | float | NYC area (~40.4–41) |
| `longitude` | float | NYC area (~−74.3–−73.6) |
| `minimum_nights` | int ≥ 1 | |
| `number_of_reviews` | int ≥ 0 | |
| `reviews_per_month` | float ≥ 0 | |
| `calculated_host_listings_count` | int ≥ 0 | |
| `availability_365` | int 0–365 | |

**Model limitations.** Trained on 2019 data; explains only part of price variation
(test R² ≈ 0.15) and is least accurate for luxury listings above $1,000/night.
Treat predictions as decision support, not appraisal.

## 🏗 Application Architecture

```
Browser ──> React + TypeScript + Tailwind (Nginx container, port 8080)
                 │  /api/* proxied by Nginx
                 ▼
            FastAPI + Uvicorn (Python container, port 8000)
                 │  pickle.load() once at startup (lifespan)
                 ▼
            models/best_price_model.pkl  (preprocessing + tuned Log-XGBoost → USD)
```

## 🔌 Backend API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | API status, model-loaded flag, model name/version, library versions |
| `/api/predict` | POST | Listing features (JSON) → estimated nightly price in USD |
| `/docs` | GET | Interactive OpenAPI/Swagger documentation |

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "neighbourhood_group": "Manhattan",
    "room_type": "Entire home/apt",
    "latitude": 40.7549,
    "longitude": -73.984,
    "minimum_nights": 3,
    "number_of_reviews": 45,
    "reviews_per_month": 1.2,
    "calculated_host_listings_count": 2,
    "availability_365": 200
  }'
```

```json
{
  "predicted_price": 259.07,
  "currency": "USD",
  "model_name": "Tuned Log-XGBoost",
  "model_version": "v1"
}
```

Invalid categories, missing fields, and out-of-range values return HTTP 422 with a
clean error message (no stack traces). If the model fails to load, `/api/predict`
returns HTTP 503 and `/health` reports `status: degraded`.

## 💻 Local Development

### Full stack with Docker (recommended)

```bash
docker compose up --build
```

- Frontend: http://localhost:8080
- Backend API + docs: http://localhost:8000/docs

### Backend only

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload        # from repo root, or set MODEL_PATH
pytest                               # run the backend test suite
```

The `MODEL_PATH` environment variable overrides the artifact location
(default: `<repo>/models/best_price_model.pkl`).

### Frontend only

```bash
cd frontend
npm install
npm run dev        # dev server with hot reload
npm run build      # production build (also runs tsc)
npm run lint
```

## 🐳 Docker

- `backend/Dockerfile` — `python:3.11-slim`, pinned dependency versions matching the
  artifact (scikit-learn 1.7.2, xgboost 3.1.2), model copied in, uvicorn ASGI server,
  container healthcheck.
- `frontend/Dockerfile` — multi-stage: Node 22 builds the Vite app, Nginx 1.27 serves
  the static bundle and proxies `/api/*` + `/health` to the backend service.
- `compose.yaml` — orchestrates both, waits for backend health before starting the
  frontend, production-oriented restart policies.

## 🔄 CI/CD (GitHub Actions)

`.github/workflows/ci-cd.yml` runs on pushes and PRs to `main`:

1. **Backend job** — ruff lint + full pytest suite **against the real saved model artifact**.
2. **Frontend job** — oxlint, TypeScript check, production build.
3. **Docker job** — builds both images (no model training; artifact comes from the repo).
4. **Publish job (main only)** — pushes SHA-tagged + `latest` images to
   **GHCR** (`ghcr.io/<owner>/airbnb-backend`, `.../airbnb-frontend`) using the default
   `GITHUB_TOKEN` with least-privilege permissions.

**Continuous deployment to a cloud host is intentionally not configured** — it will be
added after the hosting provider is selected and approved. Rollback procedure (once
deployed): redeploy the previous SHA-tagged image, which GHCR retains.

## 🔁 Reproducing Inference Without Retraining

```python
import pickle
import pandas as pd

with open("models/best_price_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

listing = pd.DataFrame([{
    "neighbourhood_group": "Brooklyn", "room_type": "Private room",
    "latitude": 40.68, "longitude": -73.95,
    "minimum_nights": 2, "number_of_reviews": 10,
    "reviews_per_month": 0.5, "calculated_host_listings_count": 1,
    "availability_365": 120,
}])
print(pipeline.predict(listing)[0])   # USD — inverse log transform is built in
```

Requires scikit-learn 1.7.2 / xgboost 3.1.2 (see `backend/requirements.txt`) — the
pickle must be loaded with compatible library versions.

## 📚 ML Research & Documentation

- `Airbnb_Price_Prediction_updated.ipynb` — full ML workflow: cleaning, EDA, pipeline,
  model comparison, log-target experiment, tuning, error analysis, persistence.
- `PROJECT_CONTEXT.md` — consolidated decisions, verified metrics, and architecture.
- The original `Airbnb_Price_Prediction.ipynb` and historical scripts are preserved
  untouched.

## 📄 License

MIT License
