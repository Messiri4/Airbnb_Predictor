/** Typed API service — all backend communication goes through here.
 *
 * API_BASE resolution:
 * - Docker/local: falls back to "/api" (Nginx proxies to the backend).
 * - Render/production: set VITE_API_BASE at build time, e.g.
 *   https://airbnb-api-xxxx.onrender.com  (no trailing slash).
 *   No secrets belong in VITE_* vars — they are baked into the public bundle.
 */
import type {
  HealthResponse,
  ListingFeatures,
  PredictionResponse,
} from "./types";

const API_BASE: string =
  (import.meta.env?.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ??
  "/api";

export class BackendUnreachableError extends Error {
  constructor(message = "Backend unreachable — it may be waking up.") {
    super(message);
    this.name = "BackendUnreachableError";
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) {
        message =
          typeof body.detail === "string"
            ? body.detail
            : JSON.stringify(body.detail);
      }
    } catch {
      /* keep default message */
    }
    if (res.status === 502 || res.status === 503 || res.status === 504) {
      // Render free tier: backend may be spinning up from sleep.
      throw new BackendUnreachableError(
        "The prediction service is starting up — this can take up to a minute on the free tier. Please try again in a moment.",
      );
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function predictPrice(
  listing: ListingFeatures,
): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(listing),
  });
  return handleResponse<PredictionResponse>(res);
}
