import type { PredictionResponse } from "../services/types";

interface Props {
  prediction: PredictionResponse | null;
  loading: boolean;
  error: string | null;
}

export default function PredictionResult({
  prediction,
  loading,
  error,
}: Props) {
  return (
    <section
      aria-live="polite"
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
    >
      <h2 className="text-lg font-semibold text-slate-900">
        Prediction Result
      </h2>

      {loading && (
        <div className="mt-6 flex items-center gap-3 text-slate-600">
          <span
            aria-hidden
            className="h-5 w-5 animate-spin rounded-full border-2 border-rose-500 border-t-transparent"
          />
          <span>Estimating nightly price…</span>
        </div>
      )}

      {!loading && error && (
        <div
          role="alert"
          className="mt-6 rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700"
        >
          <p className="font-semibold">Prediction failed</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {!loading && !error && !prediction && (
        <p className="mt-4 text-sm text-slate-500">
          Fill in the property details and submit to see an estimated nightly
          price.
        </p>
      )}

      {!loading && !error && prediction && (
        <div className="mt-5">
          <p className="text-sm text-slate-500">Estimated nightly price</p>
          <p className="mt-1 text-4xl font-bold tracking-tight text-slate-900 sm:text-5xl">
            ${prediction.predicted_price.toFixed(2)}
            <span className="ml-2 align-middle text-base font-medium text-slate-500">
              {prediction.currency} / night
            </span>
          </p>
          <dl className="mt-5 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
            <div className="rounded-lg bg-slate-50 px-4 py-3">
              <dt className="text-slate-500">Model</dt>
              <dd className="font-semibold text-slate-900">
                {prediction.model_name}
              </dd>
            </div>
            <div className="rounded-lg bg-slate-50 px-4 py-3">
              <dt className="text-slate-500">Version</dt>
              <dd className="font-semibold text-slate-900">
                {prediction.model_version}
              </dd>
            </div>
          </dl>
          <p className="mt-4 text-xs leading-relaxed text-slate-500">
            Based on a model trained on historical 2019 NYC Airbnb data. Actual
            market prices vary — treat this estimate as a reference, not a
            guarantee.
          </p>
        </div>
      )}
    </section>
  );
}
