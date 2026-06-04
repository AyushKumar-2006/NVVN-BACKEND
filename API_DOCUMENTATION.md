# NVVN Power Forecast — API Documentation

Backend API for state-wise electricity load forecasting (XGBoost 5-minute models +
Prophet 30-day models). This document is for the frontend developer.

- **Base URL:** `http://127.0.0.1:8000`
- **API prefix:** `/api/power/`
- **Interactive docs (Swagger UI):** `http://127.0.0.1:8000/api/docs`
- **OpenAPI schema:** `http://127.0.0.1:8000/api/openapi.json`
- **CORS:** enabled for all origins (dev).
- **Auth:** none.

### Conventions

- **Dates** are `YYYY-MM-DD`.
- **`state_code`** is the dropdown code returned by `GET /states/in` (these are MERIT
  codes, e.g. `CHG` = Chhattisgarh, `DL` = Delhi). The backend maps them to internal
  short codes automatically.
- **Currently trained / forecastable states:** `CHG` (Chhattisgarh) and `DL` (Delhi).
  Other states return HTTP 400 until their models are trained.
- **Error format** (Django Ninja): non-2xx responses return `{"detail": "message"}`.

---

## Endpoint summary

| # | Method | URL | Purpose |
|---|--------|-----|---------|
| 1 | GET  | `/api/power/states/in` | List selectable states (dropdown) |
| 2 | GET  | `/api/power/forecast-5min` | 5-minute load forecast for one day (288 points) |
| 3 | GET  | `/api/power/forecast-30day` | 30-day daily forecast + risk (Prophet) |
| 4 | GET  | `/api/power/forecast-compare` | Predicted vs actual for one day |
| 5 | GET  | `/api/power/forecast-accuracy` | Day-by-day accuracy report (last N days) |
| 6 | GET  | `/api/power/daily-dashboard` | Combined past actuals + future forecast |
| 7 | GET  | `/api/power/previous-predictions` | Saved daily predictions (paginated) |
| 8 | GET  | `/api/power/state-current` | Live state status from meritindia.in |
| 9 | POST | `/api/power/upload-xlsx` | Upload historical load data (XLSX/CSV) |
| 10 | POST | `/api/power/train-all-models/` | Train all ML models |

---

## 1. List states (dropdown)

**`GET /api/power/states/in`**

What it does: returns every selectable state with its dropdown code and display name.
Use this to populate the state selector.

Parameters: none.

Example response (`200`):
```json
[
  { "code": "AP",  "name": "Andhra Pradesh" },
  { "code": "CHG", "name": "Chhattisgarh" },
  { "code": "DL",  "name": "Delhi" }
]
```
(34 entries total.)

---

## 2. 5-minute forecast (single day)

**`GET /api/power/forecast-5min`**

What it does: XGBoost forecast of load at 5-minute resolution for one day (288 points),
with daily summary metrics.

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state_code` | query | yes | Dropdown code, e.g. `CHG` |
| `forecast_date` | query | no | `YYYY-MM-DD` (defaults to today) |

Example: `/api/power/forecast-5min?state_code=CHG&forecast_date=2026-07-03`

Example response (`200`):
```json
{
  "state": "CG",
  "date": "2026-07-03",
  "season": "monsoon",
  "weekday": "Friday",
  "is_weekend": false,
  "is_holiday": false,
  "energy_consumption_mu_per_day": 79.87,
  "average_load_mw": 3327.84,
  "peak_load_mw": 4024.19,
  "mape_difference_percent": null,
  "points": [
    { "datetime": "2026-07-03T00:00:00", "mw": 2721.63, "temperature": 25.4 },
    { "datetime": "2026-07-03T00:05:00", "mw": 2718.10, "temperature": 25.39 }
  ]
}
```
`points` always contains 288 entries (one per 5 minutes).

---

## 3. 30-day forecast + risk (Prophet)

**`GET /api/power/forecast-30day`**

What it does: Prophet daily forecast for 30 days starting at `from_date`, with a
predicted peak/average per day, a free weather temperature, and a High-Risk-Day
classification against the state's historical peak.

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state_code` | query | yes | Dropdown code, e.g. `CHG` |
| `from_date` | query | yes | `YYYY-MM-DD` — first day of the 30-day window |

Risk levels (predicted daily peak vs historical peak): `NORMAL`, `HIGH RISK` (≥85%),
`CRITICAL` (≥95%). Weather: Open-Meteo forecast for ~first 16 days, then climatology
(same calendar day averaged over the past 3 years).

Example: `/api/power/forecast-30day?state_code=CHG&from_date=2026-07-01`

Example response (`200`, `days` trimmed to 1 of 30):
```json
{
  "state": "CG",
  "state_code": "CG",
  "model": "prophet",
  "from_date": "2026-07-01",
  "to_date": "2026-07-30",
  "days_forecasted": 30,
  "historical_peak_mw": 5300.0,
  "risk_thresholds_mw": { "high_risk": 4505.0, "critical": 5035.0 },
  "risk_summary": { "normal_days": 30, "high_risk_days": 0, "critical_days": 0 },
  "trained_through": "2025-05-31",
  "days": [
    {
      "date": "2026-07-01",
      "weekday": "Wednesday",
      "is_weekend": false,
      "is_holiday": false,
      "predicted_peak_load_mw": 4466.37,
      "predicted_average_load_mw": 3598.32,
      "peak_pct_of_historical": 84.3,
      "risk_level": "NORMAL",
      "temperature_c": 27.57,
      "weather_source": "climatology"
    }
  ]
}
```

---

## 4. Predicted vs actual (single day)

**`GET /api/power/forecast-compare`**

What it does: compares the model's prediction for a date against the actual recorded
load (when the date exists in the database). If there is no actual data for the date
(e.g. a future date), the prediction is returned alone with a `note`.

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state_code` | query | yes | Dropdown code, e.g. `CHG` |
| `date` | query | yes | `YYYY-MM-DD` |

Example: `/api/power/forecast-compare?state_code=CHG&date=2025-05-31`

Example response (`200`, date with actual data):
```json
{
  "state": "CG",
  "state_code": "CG",
  "date": "2025-05-31",
  "actual_data_available": true,
  "predicted": { "average_load_mw": 4117.1, "peak_load_mw": 4806.16, "energy_mu": 98.81, "points": 288 },
  "actual":    { "average_load_mw": 3948.78, "peak_load_mw": 4773.28, "energy_mu": 94.77, "points": 288 },
  "difference": {
    "average": { "mw": 168.32, "pct": 4.26 },
    "peak":    { "mw": 32.88,  "pct": 0.69 },
    "energy":  { "mw": 4.04,   "pct": 4.26 }
  },
  "mape_pct": 4.45,
  "accuracy_score_pct": 95.55,
  "points_compared": 288
}
```
When no actual data exists, `actual`, `difference`, `mape_pct`, and
`accuracy_score_pct` are `null` and a `note` field is added.

---

## 5. Accuracy report (last N days)

**`GET /api/power/forecast-accuracy`**

What it does: day-by-day predicted vs actual accuracy over the most recent `days` days
that have actual data, plus an overall MAPE/accuracy and the best/worst day.

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state_code` | query | yes | Dropdown code, e.g. `CHG` |
| `days` | query | no | Number of recent days (default 30, max 120) |

Example: `/api/power/forecast-accuracy?state_code=CHG&days=30`

Example response (`200`, `daily` trimmed):
```json
{
  "state": "CG",
  "state_code": "CG",
  "days_requested": 30,
  "days_evaluated": 30,
  "date_range": { "from": "2025-05-02", "to": "2025-05-31" },
  "overall": { "mape_pct": 2.01, "accuracy_pct": 97.99 },
  "best_day":  { "date": "2025-05-19", "accuracy_pct": 98.76, "error_pct": 1.24, "predicted_avg_mw": 4291.1, "actual_avg_mw": 4288.09 },
  "worst_day": { "date": "2025-05-17", "accuracy_pct": 95.29, "error_pct": 4.71, "predicted_avg_mw": 4113.85, "actual_avg_mw": 3939.47 },
  "daily": [
    {
      "date": "2025-05-02",
      "predicted_avg_mw": 4321.02,
      "actual_avg_mw": 4291.77,
      "predicted_peak_mw": 5027.89,
      "actual_peak_mw": 5284.35,
      "error_pct": 1.54,
      "accuracy_pct": 98.46,
      "points_compared": 288
    }
  ]
}
```
Note: takes ~9s for 30 days (runs the model per day).

---

## 6. Daily dashboard (past + future combined)

**`GET /api/power/daily-dashboard`**

What it does: a single timeline of the most recent 30 days with actual data plus the
next 30 days of forecast. Past days carry actual-vs-predicted accuracy; future days
carry the Prophet forecast plus a power-requirement insight
(`how_much_needed_mw`, `shortage_risk`, `recommended_reserve_mw` = peak × 1.15).

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state_code` | query | yes | Dropdown code, e.g. `CHG` |

Example: `/api/power/daily-dashboard?state_code=CHG`

Example response (`200`, `days` trimmed to one of each type):
```json
{
  "state": "CG",
  "state_code": "CG",
  "historical_peak_mw": 5300.0,
  "past_days": 30,
  "forecast_days": 30,
  "total_days": 60,
  "days": [
    {
      "date": "2025-05-31",
      "status": "actual",
      "actual_mw": 3948.78,
      "predicted_mw": 4117.1,
      "actual_peak_mw": 4773.28,
      "predicted_peak_mw": 4806.16,
      "accuracy_pct": 95.55,
      "error_pct": 4.45
    },
    {
      "date": "2026-06-04",
      "status": "forecast",
      "predicted_load_mw": 3950.17,
      "predicted_peak_mw": 4865.94,
      "predicted_energy_mu": 94.8,
      "risk_level": "HIGH RISK",
      "temperature_c": 33.7,
      "how_much_needed_mw": 4865.94,
      "shortage_risk": false,
      "recommended_reserve_mw": 5595.83
    }
  ]
}
```
Note: takes ~14s (30 per-day model predictions + Prophet). `status` is either
`"actual"` (past) or `"forecast"` (future).

---

## 7. Previous predictions (paginated)

**`GET /api/power/previous-predictions`**

What it does: returns previously saved daily predictions (written whenever a 5-minute
forecast is generated). Paginated.

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state` | query | yes | Dropdown code, e.g. `CHG` |
| `forecast_date` | query | no | `YYYY-MM-DD` filter (defaults to today) |
| `page` | query | no | Page number (page size 10) |

Example: `/api/power/previous-predictions?state=CHG&forecast_date=2026-07-03`

Example response (`200`):
```json
{
  "items": [
    { "state": "CG", "date": "2026-07-03", "load_mw": 3327.84 }
  ],
  "count": 1
}
```

---

## 8. Live state status (MERIT India)

**`GET /api/power/state-current`**

What it does: scrapes the current demand/ISGS/import figures for a state from
meritindia.in in real time. Depends on that external site being reachable.

| Parameter | In | Required | Description |
|-----------|----|----------|-------------|
| `state` | query | yes | Dropdown code, e.g. `CHG` |

Example: `/api/power/state-current?state=CHG`

Example response (`200`):
```json
[
  { "Demand": "264", "ISGS": null, "ImportData": "264" }
]
```

---

## 9. Upload historical data

**`POST /api/power/upload-xlsx`**

What it does: uploads an XLSX/CSV of historical load and saves it to the database.
Supports a generic 5-minute format with columns `DateTime, State, Load_MW`
(also accepts the region-hourly and Delhi 5-minute formats).

Request: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | `.xlsx`, `.xls`, or `.csv` |

Example:
```bash
curl -X POST http://127.0.0.1:8000/api/power/upload-xlsx \
  -F "file=@CG_load_data.xlsx"
```

Example response (`200`):
```json
{ "status": "success", "rows_inserted": 254016, "ml_status": "retrained" }
```
Errors: `400` with `{"detail": "..."}` for unsupported columns / empty file.

---

## 10. Train all models

**`POST /api/power/train-all-models/`**

What it does: trains the 5-minute models for all states that have data (runs
synchronously in the current setup). States without data are skipped.

Parameters: none.

Example:
```bash
curl -X POST http://127.0.0.1:8000/api/power/train-all-models/
```

Example response (`200`):
```json
{ "message": "Model training has started in the background. Check logs for progress." }
```

---

## Quick reference for the frontend

- Populate the state dropdown from **#1 `/states/in`** (use the `code` as `state_code`).
- For a single-day load curve, call **#2 `/forecast-5min`** and plot `points[].mw`.
- For a month-ahead view with risk flags, call **#3 `/forecast-30day`**.
- For "how accurate is the model", call **#5 `/forecast-accuracy`** (and **#4 `/forecast-compare`** for a single day).
- For a combined past+future overview, call **#6 `/daily-dashboard`**.
- Only `CHG` and `DL` have trained models right now; other codes return `400`.
