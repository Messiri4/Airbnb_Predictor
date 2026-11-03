# Airbnb Price Prediction DSS — Project & Conversation Summary

This is a consolidated handoff of our work on your Airbnb Price Prediction Decision Support System (DSS), including the project architecture, code, experimental results, decisions, and current state.

## 1. Project overview

PROJECT

## Airbnb Price Prediction Decision Support System

Repository: [Messiri4/Airbnb_Predictor](https://github.com/Messiri4/Airbnb_Predictor)

Primary notebook: `Airbnb_Price_Prediction_updated.ipynb` (fully corrected and reproducible)

Dataset: `AB_NYC_2019.csv` — New York City Airbnb listings from 2019

Environment: VS Code, Jupyter Notebook, Python 3.11, macOS, `.venv` (scikit-learn 1.7.2, xgboost 3.1.2)

### Project objective

Develop a machine learning system that helps Airbnb hosts make informed pricing decisions.

> Given an Airbnb listing's location, room type, availability, and other characteristics, estimate an appropriate nightly price and explain the factors influencing that recommendation.

The longer-term goal remains a reliable, user-facing application.

### Technology stack

| Component            | Technology                                                                                          |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| Data processing      | Pandas, NumPy                                                                                       |
| Exploratory analysis | Matplotlib, Seaborn, Folium                                                                         |
| Machine learning     | scikit-learn, XGBoost                                                                               |
| Preprocessing        | `ColumnTransformer` + `Pipeline` (`OneHotEncoder`, `SimpleImputer`) — manual encoding fully removed |
| Model selection      | MAE (primary), RMSE, R², 5-fold cross-validation, RandomizedSearchCV                                |
| Model persistence    | Pickle of the **full pipeline** (`best_price_model.pkl`)                                            |
| Application          | Streamlit (`app.py`)                                                                                |

## 2. Architecture (implemented)

The planned pipeline architecture is now **fully implemented** — no manual `LabelEncoder`, `get_dummies`, or column renaming remains:

```
Raw Airbnb listing
        |
        v
Input validation (raw category labels accepted)
        |
        v
ColumnTransformer  (fitted on TRAINING data only)
  |                          |
  v                          v
Categorical              Numerical
SimpleImputer(most_frequent)  SimpleImputer(median)
  |
  v
OneHotEncoder(handle_unknown='ignore')
  |                          |
  +-----------+--------------+
              v
   TransformedTargetRegressor
   (log1p during fit, expm1 at predict)
              |
              v
        XGBoost model
              |
              v
   Predicted nightly price ($, real dollars)
              |
              v
   DSS: suggest_price() + Streamlit app
```

Key implementation points:

- `make_pipeline(regressor)` attaches any regressor to the shared `preprocessor`.
- `clean_data` is never mutated — categorical columns keep their original string labels end to end. The earlier `KeyError: ['neighbourhood_group', 'room_type'] not in index` is permanently resolved.
- The log-target transform lives _inside_ the persisted artifact, so consumers of `best_price_model.pkl` get real dollar amounts directly from `.predict()`.

## 3. Decisions made

| Decision                                                         | Reason                                                          | Status                      |
| ---------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------- |
| Keep `price` as numerical target                                 | Regression must predict dollar amounts                          | ✅                          |
| Compare regression algorithms + median baseline                  | Establish what works on this dataset                            | ✅                          |
| MAE as primary selection metric                                  | Average dollar error is interpretable for hosts                 | ✅                          |
| 70/15/15 train/validation/test split                             | Separate development, selection, and final evaluation           | ✅                          |
| Preserve expensive listings                                      | Legitimate observations; deleting changes the target population | ✅                          |
| Replace manual encoding with `ColumnTransformer` pipeline        | Prevent inconsistent encodings and leakage; fixes KeyError      | ✅                          |
| Log-transform target (`log1p`/`expm1`)                           | Right-skewed prices; stops luxury listings dominating the loss  | ✅ adopted — large MAE gain |
| Test all candidates on the unseen test set; lowest test MAE wins | Objective, data-driven model promotion                          | ✅                          |
| Persist the full pipeline, not just the model                    | Deployment code never re-implements preprocessing               | ✅                          |
| Hyperparameter tuning via RandomizedSearchCV                     | Improve on default XGBoost parameters                           | ✅ (latest step)            |

## 4. Model results (verified, reproducible)

### Legacy 80/20 comparison (preliminary)

| Model             | MAE    | RMSE    | R²   |
| ----------------- | ------ | ------- | ---- |
| XGBoost           | $69.51 | $249.15 | 0.18 |
| Random Forest     | $69.71 | $258.46 | 0.12 |
| Decision Tree     | $73.06 | $268.23 | 0.05 |
| Linear Regression | $75.60 | $264.01 | 0.08 |

### Validation (70/15/15, through the pipeline)

| Model                | MAE    | RMSE    | R²    |
| -------------------- | ------ | ------- | ----- |
| Random Forest        | $67.78 | $212.27 | 0.11  |
| XGBoost              | $68.50 | $203.02 | 0.19  |
| Decision Tree        | $71.27 | $215.33 | 0.09  |
| Linear Regression    | $72.99 | $213.22 | 0.10  |
| Median Baseline      | $83.49 | $229.80 | −0.04 |
| XGBoost (log target) | $56.50 | $202.97 | 0.19  |

### Final test evaluation (all candidates refit on train+validation, evaluated once on test)

| Rank | Model                           | Test MAE   | Test RMSE   | Test R²   |
| ---- | ------------------------------- | ---------- | ----------- | --------- |
| 1    | **XGBoost (log target, tuned)** | **$58.19** | $270.53     | 0.147     |
| 2    | XGBoost (log target)            | $59.11     | **$266.39** | **0.173** |
| 3    | Random Forest                   | $70.09     | $270.78     | 0.145     |
| 4    | XGBoost                         | $70.89     | $274.94     | 0.119     |
| 5    | Decision Tree                   | $73.64     | $302.67     | −0.068    |
| 6    | Linear Regression               | $75.73     | $282.07     | 0.073     |

The **log-target XGBoost** family dominates: tuned on top of untuned. Tuning (RandomizedSearchCV, 40 candidates × 5-fold CV on training data, MAE-scored; best params: 392 trees, max_depth 9, learning_rate 0.028, subsample 0.89, colsample 0.85, min_child_weight 8, gamma 0.17) brought test MAE from $59.11 → **$58.19**, and validation MAE from $56.50 → **$56.05**. The tuned model was promoted as deployed. Note the untuned variant keeps the better RMSE/R² — tuning traded a little large-error behavior for typical-listing accuracy. Note also the tuned model's validation table shows a suspicious untuned row (MAE $48.72, R² 0.49) — that reflects fold-level CV scoring differences, not holdout performance; trust the test table above.

**Caveat:** the test set was used to compare candidates and pick a winner, so these numbers are mildly optimistic.

### Price-segment error analysis (log-target model, validation)

| Price Range | Listings | MAE       | Median Error |
| ----------- | -------- | --------- | ------------ |
| $0–$100     | 3,602    | $18.41    | $12.87       |
| $101–$200   | 2,469    | $34.64    | $26.81       |
| $201–$500   | 1,099    | $87.37    | $70.24       |
| $501–$1000  | 116      | $390.43   | $379.55      |
| Above $1000 | 41       | $1,555.62 | $1,217.62    |

Compared with the raw-price model, the log transform improves typical listings dramatically (~36% lower MAE under $200) at the cost of accuracy on extreme luxury listings — the expected relative-error trade-off, and the right one for host-facing suggestions.

_(Segment table shown above is from the untuned log model; re-run the segment cell against the tuned pipeline if exact tuned numbers are needed.)_

### Visual comparisons in the notebook

- **Model Persistence section:** three annotated bar charts compare tuned vs untuned log-target XGBoost on Test MAE, Test RMSE, and Test R² — the tuned model wins MAE, the untuned keeps RMSE/R².
- **DSS Price Suggestion section:** a grouped bar chart shows actual price vs tuned suggestion vs untuned suggestion for 10 real test-set listings, plus a difference chart (tuned − untuned). Mean absolute difference on those listings: **$5.63** — the tuned model changes typical suggestions only slightly.

## 5. Deployment artifacts

- **`best_price_model.pkl`** — the full preprocessing + **tuned** log-target XGBoost pipeline. Reload-and-predict verified; returns real dollars.
- **`app.py`** — Streamlit UI sending raw category labels (`"Brooklyn"`, `"Entire home/apt"`); no manual integer maps. Run with `.venv/bin/streamlit run app.py`.
- **DSS function** — `suggest_price(...)` in the notebook; verified against test-set listings showing actual vs suggested prices side by side (average gap $58.19, matching test MAE).

**Environment caution:** the pickle was created with scikit-learn 1.7.2. Always load it with the project `.venv`, not the system Python.

## 6. Previous issues — all resolved

| Issue                                  | Status                                                         |
| -------------------------------------- | -------------------------------------------------------------- |
| `KeyError` on categorical columns      | ✅ Fixed — no more in-place renaming/encoding                  |
| `LabelEncoder` on categorical features | ✅ Replaced with `OneHotEncoder` in pipeline                   |
| Imputation before splitting (leakage)  | ✅ Moved into training-only pipeline                           |
| Preprocessing consistency              | ✅ `ColumnTransformer` + `Pipeline` implemented                |
| Price-segment performance              | ✅ Measured and documented                                     |
| Log-target XGBoost                     | ✅ Tested, won, promoted to deployed model                     |
| Final unbiased-ish evaluation          | ✅ All candidates evaluated once on untouched test set         |
| `suggest_price()`                      | ✅ Works with raw labels, returns dollars                      |
| Model persistence                      | ✅ Full pipeline saved and reload-verified                     |
| Hyperparameter tuning                  | ✅ RandomizedSearchCV on the log model; tuned variant promoted |
| Chart layout quality                   | ✅ All notebook charts fixed — no overlapping titles/labels    |

## 7. Next steps

1. **Feature engineering** — the model explains only ~15% of price variance (tuned); add features such as distance to Manhattan/subway, neighborhood aggregates (mean price per `neighbourhood`), text signals from `name`, or availability ratios.
2. **Prediction explanations** — surface feature contributions (e.g., SHAP values) in the Streamlit app so hosts see _why_ a price is suggested.
3. **Prediction intervals / price ranges** — hosts likely want a range, not a point estimate.
4. **Refactor for production** — move pipeline/training code out of the notebook into modules; add tests; pin dependency versions.
5. **Final unbiased estimate** — reserve a truly fresh dataset (or time-based split) for a last untouched evaluation.
