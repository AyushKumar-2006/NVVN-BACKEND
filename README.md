# NVVN-Backend — Chhattisgarh Power Demand Forecasting

A Django backend that forecasts Chhattisgarh (CG) electricity demand and serves it
through a multi-page dashboard. Built with Django, django-ninja, XGBoost, Prophet and Chart.js.

- **30-day forecast** — XGBoost (days 1–15) + Prophet (days 16–30) as one smooth curve.
- **Weather-aware** — population-weighted temperature from 4 districts (Raipur, Bilaspur, Korba, Jagdalpur).
- **Always current** — a daily job keeps the load and weather data up to date.

## Run

Install the requirements, run migrations, then start the server. The dashboard opens at
http://127.0.0.1:8000/dashboard/cg/ and the API docs at /api/docs.

## Dashboard

Overview · Today's Load · 30-Day Forecast · Temperature · Energy Trend — all under `/dashboard/cg/`.

Full guide: [docs/CG_FORECAST_GUIDE.md](docs/CG_FORECAST_GUIDE.md)
