/** Types mirroring the FastAPI schema (backend/app/schemas.py). */

export const NEIGHBOURHOOD_GROUPS = [
  "Bronx",
  "Brooklyn",
  "Manhattan",
  "Queens",
  "Staten Island",
] as const;

export const ROOM_TYPES = [
  "Entire home/apt",
  "Private room",
  "Shared room",
] as const;

export interface ListingFeatures {
  neighbourhood_group: string;
  room_type: string;
  latitude: number;
  longitude: number;
  minimum_nights: number;
  number_of_reviews: number;
  reviews_per_month: number;
  calculated_host_listings_count: number;
  availability_365: number;
}

export interface PredictionResponse {
  predicted_price: number;
  currency: string;
  model_name: string;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string;
  model_version: string;
  sklearn_version: string | null;
  xgboost_version: string | null;
}
