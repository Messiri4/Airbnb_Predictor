import type { HealthResponse } from "../services/types";

interface Props {
  health: HealthResponse | null;
}

const METRICS: Array<{
  label: string;
  testMae: string;
  testRmse: string;
  testR2: string;
}> = [
  {
    label: "Tuned Log-XGBoost (deployed)",
    testMae: "$58.19",
    testRmse: "$270.53",
    testR2: "0.147",
  },
];

export default function ModelInfo({ health }: Props) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-900">About the Model</h2>

      <div className="mt-4 space-y-3 text-sm leading-relaxed text-slate-600">
        <p>
          Predictions come from the{" "}
          <strong>tuned Log-XGBoost regression model</strong> trained and saved
          during this project's ML development phase. The saved artifact
          includes all preprocessing (one-hot encoding and imputation) and the
          logarithmic target transformation, so predictions are returned
          directly in USD.
        </p>
        <p>
          <strong>Training data:</strong> the New York City Airbnb Open Data
          dataset (2019), 48,895 listings across the five boroughs.
        </p>
        <p>
          <strong>Documented hold-out test performance</strong> (from the
          notebook's final evaluation, reproduced here without recalculation):
        </p>
      </div>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-2 pr-3">Model</th>
              <th className="py-2 pr-3">Test MAE</th>
              <th className="py-2 pr-3">Test RMSE</th>
              <th className="py-2">Test R²</th>
            </tr>
          </thead>
          <tbody>
            {METRICS.map((m) => (
              <tr key={m.label} className="border-t border-slate-100">
                <td className="py-2 pr-3 font-medium text-slate-900">
                  {m.label}
                </td>
                <td className="py-2 pr-3">{m.testMae}</td>
                <td className="py-2 pr-3">{m.testRmse}</td>
                <td className="py-2">{m.testR2}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {health && (
        <p className="mt-4 text-xs text-slate-500">
          Serving stack: scikit-learn {health.sklearn_version ?? "—"}, XGBoost{" "}
          {health.xgboost_version ?? "—"}
        </p>
      )}

      <div className="mt-4 rounded-lg bg-amber-50 px-4 py-3 text-xs leading-relaxed text-amber-800">
        <strong>Limitations.</strong> Trained on 2019 data — prices, market
        conditions, and regulations have changed since. The model explains only
        part of price variation (test R² ≈ 0.15) and is least accurate for
        luxury listings above $1,000/night. Treat outputs as decision support,
        not appraisal.
      </div>
    </section>
  );
}
