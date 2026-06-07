# NVVN-Backend — Chhattisgarh Power Demand Forecasting

A Django backend that forecasts Chhattisgarh (CG) electricity demand and serves it
through a multi-page dashboard. Built with Django, django-ninja, XGBoost, Prophet and Chart.js.

- **30-day forecast** — XGBoost (days 1–15) + Prophet (days 16–30) as one smooth curve.
- **Weather-aware** — population-weighted temperature from 4 districts (Raipur, Bilaspur, Korba, Jagdalpur).
- **Always current** — a daily job keeps the load and weather data up to date.

## Quick start

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Dashboard: http://127.0.0.1:8000/dashboard/cg/  ·  API docs: http://127.0.0.1:8000/api/docs

## Dashboard

`Overview` · `Today's Load` · `30-Day Forecast` · `Temperature` · `Energy Trend`
— all under `/dashboard/cg/`.

## Common commands

```bash
python manage.py retrain_cg_models     # train XGBoost + Prophet, print metrics
python manage.py backfill_cg_live      # bring load + weather up to today
python manage.py fetch_wrldc_psp --years 2026   # pull real WRLDC demand
```

Full guide: [docs/CG_FORECAST_GUIDE.md](docs/CG_FORECAST_GUIDE.md)
