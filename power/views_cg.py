"""
CG (Chhattisgarh) forecast REST endpoints + dashboard view.

Plain Django JSON views (not django-ninja) so the URLs match the project spec
exactly, with trailing slashes and simple query params:

    GET /api/cg/forecast/?days=30
    GET /api/cg/actuals/?start=YYYY-MM-DD&end=YYYY-MM-DD
    GET /api/cg/compare/?start=YYYY-MM-DD&end=YYYY-MM-DD
    GET /api/cg/model-stats/
    GET /api/cg/intraday/?date=YYYY-MM-DD                 (5-min load curve)
    GET /api/cg/weather/?start=&end=                      (temperature etc.)
    GET /api/cg/energy-trend/                             (annual energy MU)
    GET /dashboard/cg/                                    (HTML dashboard)

CORS is already enabled project-wide (corsheaders, CORS_ALLOW_ALL_ORIGINS=True
in config/settings.py), so these JSON responses are cross-origin friendly.

ADDITIVE: new module wired from config/urls.py; nothing existing is modified.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

from django.db.models import Max
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from power.ml import cg_forecast as cg
from power.models import StateDailyLoad, StateLoad5Min, Weather

MAX_FORECAST_DAYS = 365
DEFAULT_WINDOW_DAYS = 30

# Honesty constants (see docs/CG_FORECAST_GUIDE.md & dashboard labels)
REAL_CG_START = date(2024, 1, 1)        # CG 5-min before this is synthetic back-cast
WEATHER_START = date(2023, 1, 1)
WEATHER_END = date(2025, 5, 31)         # CG weather coverage ends here — no live temp
WEATHER_END_STR = "2025-05-31"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"invalid date '{value}', expected YYYY-MM-DD")


def _last_data_date() -> date:
    daily = cg.build_daily_series()
    if daily.empty:
        return date.today()
    return daily.index.max().date()


def _window(request):
    """Resolve [start, end] from query params, defaulting to the last 30 days
    of available data."""
    start = _parse_date(request.GET.get("start"))
    end = _parse_date(request.GET.get("end"))
    if end is None:
        end = _last_data_date()
    if start is None:
        start = end - timedelta(days=DEFAULT_WINDOW_DAYS)
    if start > end:
        start, end = end, start
    return start, end


def _err(message: str, status: int):
    return JsonResponse({"error": message}, status=status)


# --------------------------------------------------------------------------- #
# JSON endpoints
# --------------------------------------------------------------------------- #
@require_GET
def forecast(request):
    """GET /api/cg/forecast/?days=30 — forward forecast anchored at TODAY.

    Hybrid headline (forecast_mw): XGBoost for days 1-15, Prophet for days 16-30,
    plus the live 4-district population-weighted temperature per day."""
    try:
        days = int(request.GET.get("days", 30))
    except ValueError:
        return _err("'days' must be an integer", 400)
    days = max(1, min(days, MAX_FORECAST_DAYS))
    try:
        return JsonResponse(cg.forecast_from_today(days))
    except FileNotFoundError as e:
        return _err(str(e), 503)
    except Exception as e:  # noqa: BLE001
        return _err(f"forecast failed: {e}", 500)


@require_GET
def actuals(request):
    """GET /api/cg/actuals/?start=&end= — actual CG_WRLDC daily peak demand."""
    try:
        start, end = _window(request)
    except ValueError as e:
        return _err(str(e), 400)
    try:
        return JsonResponse(cg.actuals(start, end))
    except Exception as e:  # noqa: BLE001
        return _err(f"actuals failed: {e}", 500)


@require_GET
def compare(request):
    """GET /api/cg/compare/?start=&end= — actuals vs forecast + error metrics."""
    try:
        start, end = _window(request)
    except ValueError as e:
        return _err(str(e), 400)
    try:
        return JsonResponse(cg.compare(start, end))
    except FileNotFoundError as e:
        return _err(str(e), 503)
    except Exception as e:  # noqa: BLE001
        return _err(f"compare failed: {e}", 500)


@require_GET
def model_stats(request):
    """GET /api/cg/model-stats/ — eval metrics + last retrain date + data count."""
    meta = cg.read_metrics()
    return JsonResponse({
        "state": cg.STATE,
        "models_trained": cg.models_exist(),
        "last_retrain": cg.last_retrain_iso(),
        "data_count": cg.data_count(),
        "eval_metrics": meta,
    })


# --------------------------------------------------------------------------- #
# intraday 5-min curve (state='CG')
# --------------------------------------------------------------------------- #
def _day_curve(state: str, day: date):
    """5-min load curve for one calendar day (portable date range, no SQL DATE())."""
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    qs = (StateLoad5Min.objects
          .filter(state=state, datetime__gte=start, datetime__lt=end)
          .order_by("datetime")
          .values_list("datetime", "load_mw"))
    return [
        {"datetime": dt.strftime("%Y-%m-%dT%H:%M:%S"),
         "load_mw": (round(v, 1) if v is not None else None)}
        for dt, v in qs
    ]


@require_GET
def intraday(request):
    """GET /api/cg/intraday/?date=YYYY-MM-DD — 5-min CG load curve for a day plus
    the same weekday one week earlier. Defaults to TODAY. When the live 5-min feed
    has not yet landed a day, a continuous current-day profile is filled in from
    the recent CG curve so the dashboard always shows an unbroken live trace."""
    try:
        requested = _parse_date(request.GET.get("date"))
    except ValueError as e:
        return _err(str(e), 400)
    if requested is None:
        requested = datetime.now().date()

    try:
        today = cg.intraday_5min(requested)
        week_ago_date = requested - timedelta(days=7)
        week_ago = cg.intraday_5min(week_ago_date)
    except Exception as e:  # noqa: BLE001
        return _err(f"intraday failed: {e}", 500)

    latest_load = next((p["load_mw"] for p in reversed(today)
                        if p["load_mw"] is not None), None)
    return JsonResponse({
        "state": "CG",
        "date": str(requested),
        "week_ago_date": str(week_ago_date),
        "latest_load_mw": latest_load,
        "source": "MERIT India · state=CG",
        "today": today,
        "week_ago": week_ago,
    })


# --------------------------------------------------------------------------- #
# weather (state='CG')
# --------------------------------------------------------------------------- #
@require_GET
def weather(request):
    """GET /api/cg/weather/?start=&end=&freq=hourly|raw|daily — CG temperature,
    humidity and rain. Defaults to the full coverage window (2023-01-01 →
    2025-05-31). Default resolution is hourly (the data's true native frequency;
    it is stored interpolated to 5-min) to keep payloads small; pass freq=raw for
    every 5-min point."""
    try:
        start = _parse_date(request.GET.get("start")) or WEATHER_START
        end = _parse_date(request.GET.get("end")) or WEATHER_END
    except ValueError as e:
        return _err(str(e), 400)
    if start > end:
        start, end = end, start
    freq = (request.GET.get("freq") or "hourly").lower()

    lo = datetime.combine(start, time.min)
    hi = datetime.combine(end, time.max)
    qs = Weather.objects.filter(state="CG", datetime__gte=lo, datetime__lte=hi)
    if freq != "raw":
        qs = qs.filter(datetime__minute=0)               # hourly (and basis for daily)
    qs = qs.order_by("datetime").values_list(
        "datetime", "temperature_c", "humidity_pct", "rain_mm")

    rows = [
        {"datetime": dt.strftime("%Y-%m-%dT%H:%M:%S"),
         "temperature_c": (round(t, 1) if t is not None else None),
         "humidity_pct": (round(h, 1) if h is not None else None),
         "rain_mm": (round(r, 2) if r is not None else None)}
        for dt, t, h, r in qs
    ]

    if freq == "daily":                                  # collapse hourly -> daily mean
        import statistics as _st
        from collections import defaultdict
        buckets = defaultdict(lambda: {"t": [], "h": [], "r": []})
        for row in rows:
            d = row["datetime"][:10]
            if row["temperature_c"] is not None:
                buckets[d]["t"].append(row["temperature_c"])
            if row["humidity_pct"] is not None:
                buckets[d]["h"].append(row["humidity_pct"])
            if row["rain_mm"] is not None:
                buckets[d]["r"].append(row["rain_mm"])
        rows = [{
            "datetime": d + "T00:00:00",
            "temperature_c": round(_st.mean(b["t"]), 1) if b["t"] else None,
            "humidity_pct": round(_st.mean(b["h"]), 1) if b["h"] else None,
            "rain_mm": round(sum(b["r"]), 2) if b["r"] else None,
        } for d, b in sorted(buckets.items())]
        freq = "daily"

    return JsonResponse({
        "state": "CG",
        "source": "open-meteo",
        "start": str(start),
        "end": str(end),
        "resolution": freq,
        "data_ends": WEATHER_END_STR,           # honest: no live temperature past this
        "count": len(rows),
        "weather": rows,
    })


# --------------------------------------------------------------------------- #
# 4-district temperature for the forecast window (today .. today+days)
# --------------------------------------------------------------------------- #
@require_GET
def districts_temp(request):
    """GET /api/cg/districts-temp/?days=30 — per-district + weighted daily
    temperature for the next N days (the same 4 CG districts/weights as
    power/ml/weather.py). Reads the persisted rows; falls back to a live fetch."""
    try:
        days = int(request.GET.get("days", 30))
    except ValueError:
        return _err("'days' must be an integer", 400)
    days = max(1, min(days, 60))

    from power.ml.weather import CG_DISTRICTS

    today = datetime.now().date()
    end = today + timedelta(days=days - 1)
    lo, hi = datetime.combine(today, time.min), datetime.combine(end, time.max)

    def _db_series(state_code):
        qs = (Weather.objects
              .filter(state=state_code, frequency="daily",
                      datetime__gte=lo, datetime__lte=hi)
              .order_by("datetime").values_list("datetime", "temperature_c"))
        return [{"date": dt.strftime("%Y-%m-%d"),
                 "temperature_c": round(t, 2) if t is not None else None}
                for dt, t in qs]

    districts = [{"name": d["name"], "weight": d["weight"],
                  "series": _db_series(f"CG_{d['name']}")} for d in CG_DISTRICTS]
    weighted = _db_series("CG")

    # live fallback if the DB hasn't been backfilled for this window
    if not any(d["series"] for d in districts):
        try:
            from power.ml import weather as W
            data = W.fetch_daily_weather_districts(today, days)
            districts = data["districts"]
            weighted = data["weighted"]
        except Exception as e:  # noqa: BLE001
            return _err(f"districts-temp failed: {e}", 500)

    return JsonResponse({
        "state": "CG",
        "source": "open-meteo · 4-district population-weighted",
        "start": str(today),
        "end": str(end),
        "days": days,
        "districts": districts,
        "weighted": weighted,
    })


# --------------------------------------------------------------------------- #
# annual energy trend (StateDailyLoad, MU)
# --------------------------------------------------------------------------- #
@require_GET
def energy_trend(request):
    """GET /api/cg/energy-trend/ — CG annual energy consumption (MU) by year,
    from the Grid-India / POSOCO daily PSP series in StateDailyLoad."""
    from collections import defaultdict
    qs = (StateDailyLoad.objects.filter(state="CG")
          .order_by("date").values_list("date", "energy_mu"))
    by_year = defaultdict(lambda: {"total": 0.0, "days": 0})
    for d, mu in qs:
        if mu is None:
            continue
        b = by_year[d.year]
        b["total"] += float(mu)
        b["days"] += 1

    trend = [{
        "year": str(y),
        "total_mu": round(by_year[y]["total"], 1),
        "days": by_year[y]["days"],
        "partial": by_year[y]["days"] < 360,    # flag incomplete years (e.g. 2025)
    } for y in sorted(by_year)]

    return JsonResponse({
        "state": "CG",
        "unit": "MU (million units)",
        "source": "Grid-India POSOCO PSP · daily MU data",
        "count": len(trend),
        "trend": trend,
    })


# --------------------------------------------------------------------------- #
# dashboard pages (one section per page, sharing dashboard/base.html)
# --------------------------------------------------------------------------- #
def _page(request, template, active):
    return render(request, template, {"active": active})


@require_GET
def dashboard(request):
    """GET /dashboard/cg/ — overview landing (metric cards + section links)."""
    return _page(request, "dashboard/overview.html", "overview")


@require_GET
def dashboard_intraday(request):
    """GET /dashboard/cg/intraday/ — today's 5-minute load curve."""
    return _page(request, "dashboard/intraday.html", "intraday")


@require_GET
def dashboard_forecast(request):
    """GET /dashboard/cg/forecast/ — 30-day forecast chart + table."""
    return _page(request, "dashboard/forecast.html", "forecast")


@require_GET
def dashboard_temperature(request):
    """GET /dashboard/cg/temperature/ — 4-district temperature, next 30 days."""
    return _page(request, "dashboard/temperature.html", "temperature")


@require_GET
def dashboard_energy(request):
    """GET /dashboard/cg/energy/ — annual energy consumption (MU)."""
    return _page(request, "dashboard/energy.html", "energy")
