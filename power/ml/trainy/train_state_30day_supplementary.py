"""
Supplementary 30-day Prophet training that ALSO uses StateDailyLoad.

The existing trainer (`train_state_30day.py`) learns daily peak/avg load (MW)
from `StateLoad5Min`. That table only holds the recent (synthetic) history.
`StateDailyLoad` can hold a much longer real history (e.g. POSOCO daily energy),
but as **energy_mu (MU/day)**, not load (MW).

This module is purely additive:
  - It does NOT modify the existing trainer; it imports its building blocks
    (`_daily_aggregates`, `_build_prophet`) read-only.
  - It converts `StateDailyLoad.energy_mu` -> an estimated daily avg/peak load
    and MERGES it with the real 5-minute-derived daily series (preferring the
    real 5-min values wherever both exist), giving Prophet a longer history for
    better long-term trend/seasonality learning.
  - It returns the exact same bundle format the predictor already loads, so
    regenerating `state_30day_<STATE>.pkl` with it is a drop-in upgrade.

Energy -> load conversion
-------------------------
  avg_load_mw = energy_mu * 1000 (MWh) / 24 h        # 1 MU = 1 GWh = 1000 MWh
  peak_mw     = avg_load_mw * peak_to_avg_ratio      # ratio learned from real
                                                     # 5-min data, else default

How to run (regenerate the CG model using the supplementary source)
-------------------------------------------------------------------
    python -c "import os,django; \
        os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings'); django.setup(); \
        from power.ml.trainy.train_state_30day_supplementary import train_and_save_supplemented; \
        print(train_and_save_supplemented('CG'))"
"""

import logging

import numpy as np
import pandas as pd

from power.ml.model_store import save_model
from power.ml.trainy.train_state_30day import _build_prophet, _daily_aggregates
from power.models import StateDailyLoad
from power.utils.logger import get_logger

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Fallback peak/avg ratio (~0.8 load factor) when there is no 5-min data to
# learn a state-specific ratio from.
DEFAULT_PEAK_TO_AVG = 1.25


def _statedaily_aggregates(state: str, peak_to_avg: float) -> pd.DataFrame:
    """StateDailyLoad.energy_mu -> estimated daily avg & peak load (MW)."""
    rows = pd.DataFrame(
        StateDailyLoad.objects.filter(state=state).values("date", "energy_mu")
    )
    if rows.empty:
        return pd.DataFrame(columns=["ds", "peak", "avg", "source"])

    rows["ds"] = pd.to_datetime(rows["date"])
    rows["energy_mu"] = pd.to_numeric(rows["energy_mu"], errors="coerce")
    rows = rows.dropna(subset=["energy_mu"])
    rows = rows[rows["energy_mu"] > 0]
    if rows.empty:
        return pd.DataFrame(columns=["ds", "peak", "avg", "source"])

    rows["avg"] = rows["energy_mu"] * 1000.0 / 24.0
    rows["peak"] = rows["avg"] * peak_to_avg
    rows["source"] = "statedaily"
    return rows[["ds", "peak", "avg", "source"]].sort_values("ds")


def _learn_peak_to_avg(five: pd.DataFrame) -> float:
    """Average peak/avg ratio from the real 5-min daily aggregates."""
    if five.empty:
        return DEFAULT_PEAK_TO_AVG
    r = five["peak"] / five["avg"].replace(0, np.nan)
    r = r[np.isfinite(r)]
    if r.empty:
        return DEFAULT_PEAK_TO_AVG
    ratio = float(r.mean())
    return ratio if 0.5 < ratio < 5 else DEFAULT_PEAK_TO_AVG


def _merged_daily(state: str):
    """Merge real 5-min daily aggregates with StateDailyLoad-derived ones."""
    try:
        five = _daily_aggregates(state).copy()
        five["source"] = "fivemin"
    except Exception:
        five = pd.DataFrame(columns=["ds", "peak", "avg", "source"])

    ratio = _learn_peak_to_avg(five)
    supp = _statedaily_aggregates(state, ratio)

    if five.empty and supp.empty:
        raise ValueError(
            f"No training data in StateLoad5Min or StateDailyLoad for {state}"
        )

    combined = pd.concat([five, supp], ignore_index=True)
    # Prefer the real 5-min value on any date present in both sources.
    combined["_prio"] = (combined["source"] == "fivemin").astype(int)
    combined = (
        combined.sort_values(["ds", "_prio"])
        .drop_duplicates("ds", keep="last")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    return combined[["ds", "peak", "avg", "source"]], ratio


def train_state_30day_model_supplemented(state: str) -> dict:
    logger = get_logger(f"TRAIN-30D-SUPP-{state}")

    daily, ratio = _merged_daily(state)
    n_real = int((daily["source"] == "fivemin").sum())
    n_supp = int((daily["source"] == "statedaily").sum())
    logger.info(
        "Merged daily rows=%d (5min=%d, statedaily=%d) for %s",
        len(daily), n_real, n_supp, state,
    )

    if len(daily) < 30:
        raise ValueError(
            f"Not enough merged daily history for {state} ({len(daily)} days)"
        )

    m_peak = _build_prophet()
    m_peak.fit(daily[["ds", "peak"]].rename(columns={"peak": "y"}))

    m_avg = _build_prophet()
    m_avg.fit(daily[["ds", "avg"]].rename(columns={"avg": "y"}))

    return {
        "model": "prophet",
        "state": state,
        "peak": m_peak,
        "avg": m_avg,
        "historical_peak": float(daily["peak"].max()),
        "historical_avg": float(daily["avg"].mean()),
        "trained_from": daily["ds"].min().date().isoformat(),
        "trained_through": daily["ds"].max().date().isoformat(),
        "train_days": int(len(daily)),
        # supplementary metadata
        "supplemented": n_supp > 0,
        "fivemin_days": n_real,
        "supplementary_days": n_supp,
        "peak_to_avg_ratio": round(ratio, 3),
    }


def train_and_save_supplemented(state: str) -> dict:
    """Train with the supplementary source and overwrite state_30day_<STATE>.pkl."""
    bundle = train_state_30day_model_supplemented(state)
    save_model(f"state_30day_{state}.pkl", bundle)
    return bundle
