# CG (Chhattisgarh) Demand Forecast — Guide

End-to-end guide to the Chhattisgarh power-demand forecasting system: data,
retraining, the REST API, the live dashboard, and the daily cron refresh.

---

## 1. What it forecasts

The real Chhattisgarh series lives in `StateLoad5Min (state='CG_WRLDC')`. It is
**sparse** — about 3 points per day from the WRLDC PSP daily report (an off-peak
~03:00 reading, an evening ~18–19:00 reading, and the day's maximum-demand
point). It carries no continuous 5-minute curve and no weather columns.

So the models forecast the **daily peak load (MW)** — the day's maximum demand —
which is the canonical demand-planning metric and the most consistently present
point in the report.

**Two models**, trained on the same daily series with a chronological 80/20
train/test split:

| Model | Library | Notes |
|-------|---------|-------|
| XGBoost | `xgboost.XGBRegressor` | Gradient-boosted trees on the feature set below. Forecasts are rolled forward recursively. |
| Prophet | `prophet.Prophet` | Weekly + yearly seasonality. Native future dataframe + uncertainty band. |

**Feature set** (engineered in `power/ml/cg_forecast.py`):
`hour`, `dayofweek`, `month`, `is_weekend`, `lag_1d`, `lag_7d`, `rolling_mean_7d`.

Artifacts are written to `power/ml/models/`:
`cg_xgb.joblib`, `cg_prophet.joblib`, `eval_metrics.json`.

---

## 2. Update the data

Pull the latest WRLDC PSP reports for a year and upsert real CG demand points
(safe to re-run — it upserts and never deletes):

```bash
python manage.py fetch_wrldc_psp --years 2026
# multiple / historical years:
python manage.py fetch_wrldc_psp --years 2024,2025,2026
# everything the server publishes:
python manage.py fetch_wrldc_psp --all
```

---

## 3. Retrain the models

```bash
python manage.py retrain_cg_models
# hold out a different test fraction:
python manage.py retrain_cg_models --test-frac 0.25
```

This loads all `CG_WRLDC` rows, builds the daily peak series, feature-engineers,
trains both models on the 80% train split, prints MAE / RMSE / MAPE on the 20%
test split, overwrites the model artifacts, and writes `eval_metrics.json`.

Typical output:

```
=== Test-set metrics (daily peak MW) ===
  model            MAE      RMSE    MAPE %
  ----------------------------------------
  xgboost       196.00    272.55     3.351
  prophet       339.18    417.09     6.147
  best by RMSE : xgboost
```

---

## 4. REST API

CORS is enabled project-wide, so these are safe to call cross-origin. All return
JSON. Base path: `/api/cg/`.

### `GET /api/cg/forecast/?days=30`
Next *N* days (1–365, default 30) of daily-peak forecast from both models.

```json
{
  "state": "CG_WRLDC",
  "target": "daily_peak_load_mw",
  "days": 30,
  "generated_at": "2026-06-07T02:00:04",
  "forecast": [
    { "date": "2026-06-06", "xgboost_mw": 5660.4, "prophet_mw": 5716.3,
      "prophet_lower_mw": 5300.2, "prophet_upper_mw": 6123.1 }
  ]
}
```

### `GET /api/cg/actuals/?start=YYYY-MM-DD&end=YYYY-MM-DD`
Actual daily-peak demand (real readings only) in the window. Defaults to the last
30 days of available data if params are omitted.

```json
{
  "state": "CG_WRLDC", "metric": "daily_peak_load_mw",
  "start": "2026-05-28", "end": "2026-06-02", "count": 6,
  "actuals": [
    { "date": "2026-05-28", "peak_mw": 5966.0, "peak_hour": 0 }
  ]
}
```

### `GET /api/cg/compare/?start=YYYY-MM-DD&end=YYYY-MM-DD`
Actuals vs both models side-by-side over a historical window, plus error metrics.
(In-window backtest — XGBoost predictions use the real observed lags.) Defaults to
the last 30 days.

```json
{
  "state": "CG_WRLDC", "start": "2026-05-06", "end": "2026-06-05", "count": 31,
  "rows": [
    { "date": "2026-05-06", "actual_mw": 5474.0, "xgboost_mw": 5946.3, "prophet_mw": 5792.7 }
  ],
  "metrics": {
    "xgboost": { "mae": 187.64, "rmse": 254.86, "mape": 3.26, "n": 30 },
    "prophet": { "mae": 228.69, "rmse": 253.93, "mape": 3.901, "n": 30 }
  }
}
```

### `GET /api/cg/model-stats/`
Eval metrics (the contents of `eval_metrics.json`) plus last retrain time and the
current data count.

```json
{
  "state": "CG_WRLDC",
  "models_trained": true,
  "last_retrain": "2026-06-07T01:58:09",
  "data_count": 7503,
  "eval_metrics": { "...": "full eval_metrics.json contents" }
}
```

**Errors:** invalid date → `400`; models not yet trained → `503` with a hint to
run `retrain_cg_models`.

---

## 5. Live dashboard

Open: **http://127.0.0.1:8000/dashboard/cg/**

- Actual vs XGBoost vs Prophet line chart for a comparison window
- Metric cards: MAE, RMSE, MAPE (+ Prophet MAPE), last retrained, total data points
- 30-day rolling forecast chart (selectable 14/30/60/90-day horizon) with Prophet band
- Date-range picker to change the comparison window (**Apply**)
- **↻ Refresh all** button + automatic refresh every 24 hours
- Dark, self-contained UI (only Chart.js via CDN; no CSS frameworks)

Template: `power/templates/dashboard/cg_forecast.html`; view + URL in
`power/views_cg.py` / `config/urls.py`.

---

## 6. Daily cron refresh

`scripts/fetch_daily.sh` activates the project environment (auto-detects a
`venv`/`.venv`, a conda env via `$NVVN_CONDA_ENV`, or `$NVVN_PYTHON`) and runs
`fetch_wrldc_psp` for the current year, logging to `logs/fetch_daily.log`.

Add to your crontab (`crontab -e`) to run every day at 02:00:

```cron
0 2 * * * bash ~/NVVN-backend/scripts/fetch_daily.sh
```

Get the exact line for **this** machine (absolute paths + the right interpreter),
or install it automatically:

```bash
python manage.py setup_cron            # print the crontab line
python manage.py setup_cron --hour 3   # run at 03:00
python manage.py setup_cron --install  # append it to your crontab
```

A good operational pattern is to retrain weekly after the data refresh, e.g.:

```cron
0 2 * * *  bash ~/NVVN-backend/scripts/fetch_daily.sh
30 3 * * 0 cd ~/NVVN-backend && python manage.py retrain_cg_models   # Sundays
```

---

## 7. Dependencies

No new packages are required — everything used here (`xgboost`, `prophet`,
`joblib`, `numpy`, `pandas`, `django`, `django-ninja`, `django-cors-headers`) is
already pinned in `requirements.txt`.

---

## 8. File map

| File | Purpose |
|------|---------|
| `power/ml/cg_forecast.py` | Shared pipeline: data → features → train → forecast / compare / actuals |
| `power/management/commands/retrain_cg_models.py` | Retrain + evaluate + save |
| `power/management/commands/setup_cron.py` | Print/install the crontab line |
| `power/views_cg.py` | `/api/cg/*` JSON endpoints + dashboard view |
| `power/templates/dashboard/cg_forecast.html` | Live dashboard |
| `scripts/fetch_daily.sh` | Daily data-refresh runner (for cron) |
| `power/ml/models/cg_xgb.joblib` · `cg_prophet.joblib` · `eval_metrics.json` | Saved artifacts |
