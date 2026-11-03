import { useCallback, useEffect, useState } from "react";
import ModelInfo from "./components/ModelInfo";
import PredictionResult from "./components/PredictionResult";
import PropertyForm from "./components/PropertyForm";
import { fetchHealth, predictPrice } from "./services/api";
import type {
  HealthResponse,
  ListingFeatures,
  PredictionResponse,
} from "./services/types";

export default function App() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleSubmit = useCallback(async (features: ListingFeatures) => {
    setLoading(true);
    setError(null);
    try {
      const result = await predictPrice(features);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }, []);

  const degraded = health && !health.model_loaded;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-5 sm:px-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-xl font-bold tracking-tight sm:text-2xl">
              🏡 Airbnb Pricing Intelligence
            </h1>
            <p className="text-sm text-slate-500">
              Decision Support System — NYC listings
            </p>
          </div>
          {health && (
            <span
              className={`inline-flex items-center gap-2 self-start rounded-full px-3 py-1 text-xs font-medium sm:self-center ${
                degraded
                  ? "bg-amber-100 text-amber-800"
                  : "bg-emerald-100 text-emerald-800"
              }`}
            >
              <span
                aria-hidden
                className={`h-2 w-2 rounded-full ${
                  degraded ? "bg-amber-500" : "bg-emerald-500"
                }`}
              />
              {degraded
                ? "Model unavailable"
                : `API online · ${health.model_name}`}
            </span>
          )}
        </div>
      </header>

      <main className="mx-auto grid max-w-6xl grid-cols-1 gap-6 px-4 py-8 sm:px-6 lg:grid-cols-2">
        <div className="space-y-6">
          <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-900">
              Property Details
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Provide the listing characteristics used by the pricing model.
            </p>
            <div className="mt-5">
              <PropertyForm onSubmit={handleSubmit} loading={loading} />
            </div>
          </section>
        </div>

        <div className="space-y-6">
          <PredictionResult
            prediction={prediction}
            loading={loading}
            error={error}
          />
          <ModelInfo health={health} />
        </div>
      </main>

      <footer className="pb-8 text-center text-xs text-slate-400">
        Built with the saved tuned Log-XGBoost artifact — no models are
        retrained at runtime.
      </footer>
    </div>
  );
}
