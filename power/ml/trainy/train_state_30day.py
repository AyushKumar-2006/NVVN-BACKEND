"""
30-day-ahead daily load forecasting with Facebook Prophet (free / open source).

Two Prophet models are trained on the daily aggregates of the 5-minute load
history and bundled together with metadata:
    - peak  : daily peak load  (max of 5-min loads per day)
    - avg   : daily average load (mean of 5-min loads per day)

Prophet automatically captures yearly + weekly seasonality, trend, and Indian
public holidays. The bundle is saved as ``state_30day_<STATE>.pkl``.
"""

import logging

import pandas as pd
from prophet import Prophet

from power.models import StateLoad5Min
from power.utils.logger import get_logger

# Prophet / cmdstanpy are extremely chatty — keep their logs quiet.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


def _build_prophet():
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
    )
    # Free, open-source Indian holiday calendar (holidays package).
    m.add_country_holidays(country_name="IN")
    return m


def _daily_aggregates(state: str) -> pd.DataFrame:
    raw = pd.DataFrame(
        StateLoad5Min.objects
        .filter(state=state)
        .values("datetime", "load_mw")
        .order_by("datetime")
    )

    if raw.empty:
        raise ValueError(f"No data for state={state}")

    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["load_mw"] = pd.to_numeric(raw["load_mw"], errors="coerce")
    raw = raw.dropna(subset=["load_mw"])

    if raw.empty:
        raise ValueError(f"No usable load values for state={state}")

    raw["date"] = raw["datetime"].dt.normalize()

    daily = (
        raw.groupby("date")["load_mw"]
        .agg(peak="max", avg="mean")
        .reset_index()
        .rename(columns={"date": "ds"})
        .sort_values("ds")
    )
    return daily


def train_state_30day_model(state: str) -> dict:
    logger = get_logger(f"TRAIN-30D-{state}")
    logger.info("Building daily aggregates for %s", state)

    daily = _daily_aggregates(state)

    if len(daily) < 30:
        raise ValueError(
            f"Not enough daily history to train 30-day model for {state} "
            f"({len(daily)} days)"
        )

    logger.info("Training Prophet PEAK model (%d days)", len(daily))
    m_peak = _build_prophet()
    m_peak.fit(daily[["ds", "peak"]].rename(columns={"peak": "y"}))

    logger.info("Training Prophet AVG model (%d days)", len(daily))
    m_avg = _build_prophet()
    m_avg.fit(daily[["ds", "avg"]].rename(columns={"avg": "y"}))

    bundle = {
        "model": "prophet",
        "state": state,
        "peak": m_peak,
        "avg": m_avg,
        "historical_peak": float(daily["peak"].max()),
        "historical_avg": float(daily["avg"].mean()),
        "trained_from": daily["ds"].min().date().isoformat(),
        "trained_through": daily["ds"].max().date().isoformat(),
        "train_days": int(len(daily)),
    }

    logger.info(
        "Trained 30-day model for %s | historical_peak=%.2f MW",
        state, bundle["historical_peak"],
    )
    return bundle
