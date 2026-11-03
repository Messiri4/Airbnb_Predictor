/** Typed API service — all backend communication goes through here. */
import type {
  HealthResponse,
  ListingFeatures,
  PredictionResponse,
} from "./types";

const API_BASE: string =
  (import.meta.env?.VITE_API_BASE as string | undefined) ?? "/api";

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
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/health");
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
