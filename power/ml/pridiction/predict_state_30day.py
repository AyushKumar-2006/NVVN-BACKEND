"""
30-day-ahead forecast prediction + High-Risk-Day detection.

Loads the bundled Prophet models (peak + avg), produces a daily forecast for
30 days starting at ``from_date``, attaches a free weather temperature for each
day, and classifies each day's risk against the state's historical peak:

    predicted peak >= 95% of historical peak  -> CRITICAL
    predicted peak >= 85% of historical peak  -> HIGH RISK
    otherwise                                 -> NORMAL
"""

import logging
from datetime import date, timedelta

import pandas as pd
from ninja.errors import HttpError

from power.ml.model_store import load_model, save_model
from power.ml.trainy.train_state_30day import train_state_30day_model
from power.ml.weather import fetch_daily_weather
from power.utils.logger import get_logger
from power.utils.metadata import STATE_CODE_TO_NAME, day_metadata

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logger = get_logger("PREDICT-30D")

HIGH_RISK_FRACTION = 0.85
CRITICAL_FRACTION = 0.95
FORECAST_DAYS = 30


def load_or_train_30day(state: str) -> dict:
    """Load the saved bundle, training + saving it on first use."""
    filename = f"state_30day_{state}.pkl"
    try:
        return load_model(filename)
    except FileNotFoundError:
        logger.warning("30-day model missing for %s — training now", state)
        bundle = train_state_30day_model(state)
        save_model(filename, bundle)
        return bundle


def _risk_level(peak_pred: float, historical_peak: float):
    if historical_peak <= 0:
        return "NORMAL", 0.0
    frac = peak_pred / historical_peak
    if frac >= CRITICAL_FRACTION:
        level = "CRITICAL"
    elif frac >= HIGH_RISK_FRACTION:
        level = "HIGH RISK"
    else:
        level = "NORMAL"
    return level, round(frac * 100, 1)


def predict_state_30day_data(state: str, from_date) -> dict:
    if isinstance(from_date, str):
        from_date = date.fromisoformat(from_date.strip())
    elif not isinstance(from_date, date):
        raise ValueError("from_date must be a str or date")

    bundle = load_or_train_30day(state)
    m_peak = bundle["peak"]
    m_avg = bundle["avg"]
    historical_peak = float(bundle["historical_peak"])

    # ---- 30-day daily forecast grid ----
    dates = [from_date + timedelta(days=i) for i in range(FORECAST_DAYS)]
    future = pd.DataFrame({"ds": pd.to_datetime(dates)})

    peak_pred = m_peak.predict(future)["yhat"].clip(lower=0).to_numpy()
    avg_pred = m_avg.predict(future)["yhat"].clip(lower=0).to_numpy()

    # ---- free weather (forecast for ~16d, climatology for the rest) ----
    weather = fetch_daily_weather(state, from_date, days=FORECAST_DAYS)

    days = []
    counts = {"NORMAL": 0, "HIGH RISK": 0, "CRITICAL": 0}

    for i, d in enumerate(dates):
        peak = round(float(peak_pred[i]), 2)
        avg = round(float(avg_pred[i]), 2)
        level, pct = _risk_level(peak, historical_peak)
        counts[level] += 1
        meta = day_metadata(d)

        days.append({
            "date": d.isoformat(),
            "weekday": meta["weekday"],
            "is_weekend": meta["is_weekend"],
            "is_holiday": meta["is_holiday"],
            "predicted_peak_load_mw": peak,
            "predicted_average_load_mw": avg,
            "peak_pct_of_historical": pct,
            "risk_level": level,
            "temperature_c": weather[i]["temperature_c"],
            "weather_source": weather[i]["source"],
        })

    return {
        "state": STATE_CODE_TO_NAME.get(state, state),
        "state_code": state,
        "model": "prophet",
        "from_date": from_date.isoformat(),
        "to_date": dates[-1].isoformat(),
        "days_forecasted": FORECAST_DAYS,
        "historical_peak_mw": round(historical_peak, 2),
        "risk_thresholds_mw": {
            "high_risk": round(historical_peak * HIGH_RISK_FRACTION, 2),
            "critical": round(historical_peak * CRITICAL_FRACTION, 2),
        },
        "risk_summary": {
            "normal_days": counts["NORMAL"],
            "high_risk_days": counts["HIGH RISK"],
            "critical_days": counts["CRITICAL"],
        },
        "trained_through": bundle.get("trained_through"),
        "days": days,
    }


def build_30day_forecast_response(state: str, from_date) -> dict:
    """Thin wrapper that surfaces failures as clean HTTP 400 errors."""
    try:
        return predict_state_30day_data(state=state, from_date=from_date)
    except HttpError:
        raise
    except Exception as e:
        raise HttpError(400, str(e))
