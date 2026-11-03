import { useState } from "react";
import type { ListingFeatures } from "../services/types";
import { NEIGHBOURHOOD_GROUPS, ROOM_TYPES } from "../services/types";

interface Props {
  onSubmit: (features: ListingFeatures) => void;
  loading: boolean;
}

const labelCls = "block text-sm font-medium text-slate-700 mb-1";
const inputCls =
  "w-full rounded-lg border border-slate-300 px-3 py-2.5 text-slate-900 " +
  "focus:border-rose-500 focus:ring-2 focus:ring-rose-200 outline-none transition";

const NUMERIC_FIELDS: Array<{
  name: keyof ListingFeatures;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  hint?: string;
}> = [
  {
    name: "latitude",
    label: "Latitude",
    min: 40.4,
    max: 41,
    step: 0.0001,
    defaultValue: 40.7128,
  },
  {
    name: "longitude",
    label: "Longitude",
    min: -74.3,
    max: -73.6,
    step: 0.0001,
    defaultValue: -74.006,
  },
  {
    name: "minimum_nights",
    label: "Minimum Nights",
    min: 1,
    max: 365,
    step: 1,
    defaultValue: 3,
  },
  {
    name: "number_of_reviews",
    label: "Number of Reviews",
    min: 0,
    max: 1200,
    step: 1,
    defaultValue: 12,
  },
  {
    name: "reviews_per_month",
    label: "Reviews per Month",
    min: 0,
    max: 30,
    step: 0.1,
    defaultValue: 1.2,
  },
  {
    name: "calculated_host_listings_count",
    label: "Host Listings Count",
    min: 1,
    max: 400,
    step: 1,
    defaultValue: 1,
  },
  {
    name: "availability_365",
    label: "Availability (days/year)",
    min: 0,
    max: 365,
    step: 1,
    defaultValue: 180,
  },
];

export default function PropertyForm({ onSubmit, loading }: Props) {
  const [group, setGroup] = useState<string>("Manhattan");
  const [roomType, setRoomType] = useState<string>("Entire home/apt");
  const [numerics, setNumerics] = useState<Record<string, number>>(() =>
    Object.fromEntries(NUMERIC_FIELDS.map((f) => [f.name, f.defaultValue])),
  );
  const [error, setError] = useState<string | null>(null);

  function updateNumeric(name: string, raw: string) {
    const value = raw === "" ? NaN : Number(raw);
    setNumerics((prev) => ({ ...prev, [name]: value }));
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    for (const f of NUMERIC_FIELDS) {
      const v = numerics[f.name];
      if (Number.isNaN(v) || v < f.min || v > f.max) {
        setError(`${f.label} must be between ${f.min} and ${f.max}.`);
        return;
      }
    }
    onSubmit({
      neighbourhood_group: group,
      room_type: roomType,
      ...numerics,
    } as ListingFeatures);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-5"
      aria-label="Property details form"
      noValidate
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label htmlFor="neighbourhood_group" className={labelCls}>
            Neighbourhood Group
          </label>
          <select
            id="neighbourhood_group"
            className={inputCls}
            value={group}
            onChange={(e) => setGroup(e.target.value)}
          >
            {NEIGHBOURHOOD_GROUPS.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="room_type" className={labelCls}>
            Room Type
          </label>
          <select
            id="room_type"
            className={inputCls}
            value={roomType}
            onChange={(e) => setRoomType(e.target.value)}
          >
            {ROOM_TYPES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {NUMERIC_FIELDS.map((f) => (
          <div key={f.name}>
            <label htmlFor={f.name} className={labelCls}>
              {f.label}
            </label>
            <input
              id={f.name}
              name={f.name}
              type="number"
              inputMode="decimal"
              className={inputCls}
              min={f.min}
              max={f.max}
              step={f.step}
              value={numerics[f.name]}
              onChange={(e) => updateNumeric(f.name, e.target.value)}
            />
          </div>
        ))}
      </div>

      {error && (
        <p
          role="alert"
          className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700"
        >
          {error}
        </p>
      )}

      <button
        type="submit"
        disabled={loading}
        className="w-full rounded-xl bg-rose-600 px-6 py-3.5 font-semibold text-white
                   shadow-sm transition hover:bg-rose-700 focus:outline-none
                   focus:ring-2 focus:ring-rose-400 focus:ring-offset-2
                   disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
      >
        {loading ? "Estimating…" : "Estimate Nightly Price"}
      </button>
    </form>
  );
}
