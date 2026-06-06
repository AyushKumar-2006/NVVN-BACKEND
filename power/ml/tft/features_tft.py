"""
Feature engineering for the CG 5-min TFT model.

This mirrors the XGBoost 5-min pipeline (see power/ml/trainy/train_state_5min.py
and power/ml/pridiction/predict_state_5min.py) so that the TFT consumes the
*same* inputs: weather, calendar, peak/interaction features, the daily load
profile and seasonal lags.

Nothing here mutates existing modules — `add_calendar_features` is imported
read-only; the peak/interaction logic is re-implemented identically so the TFT
package stays self-contained.
"""

import numpy as np
import pandas as pd

from power.models import StateLoad5Min, Weather
from power.utils.metadata import add_calendar_features
from power.ml.tft.config import STEPS_PER_DAY


# ----------------------------------------------------------------------
# PEAK + INTERACTION FEATURES  (identical to power/ml/trainy/train_state_5min)
# ----------------------------------------------------------------------
def add_peak_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    df["is_peak"] = df["hour"].between(18, 23).astype(int)

    df["temp_x_hour"] = df["temperature_c"] * df["hour"]
    df["humidity_x_hour"] = df["humidity_pct"] * df["hour"]
    df["wind_x_hour"] = df["wind_speed_ms"] * df["hour"]
    return df


# ----------------------------------------------------------------------
# RAW LOAD -> CONTINUOUS 5-MIN TARGET
# ----------------------------------------------------------------------
def load_state_target(state: str) -> pd.DataFrame:
    """Return a continuous 5-min frame with columns ['ds', 'y'] for `state`."""
    raw = pd.DataFrame(
        StateLoad5Min.objects
        .filter(state=state)
        .values("datetime", "load_mw", "brpl", "bypl", "ndpl", "ndmc", "mes")
        .order_by("datetime")
    )
    if raw.empty:
        raise ValueError(f"No 5-min data for state={state}")

    raw["datetime"] = pd.to_datetime(raw["datetime"])

    discoms = ["brpl", "bypl", "ndpl", "ndmc", "mes"]
    raw["y"] = raw["load_mw"]
    raw.loc[raw["y"].isna(), "y"] = raw[discoms].sum(axis=1)

    # Only the numeric target survives the resample (discom cols are all-NULL for
    # states like CG and would break .interpolate()).
    df = (
        raw.set_index("datetime")[["y"]]
        .resample("5min")
        .mean()
        .interpolate()
        .reset_index()
        .rename(columns={"datetime": "ds"})
    )
    return df


# ----------------------------------------------------------------------
# WEATHER  (from DB, same source the XGBoost trainer persists)
# ----------------------------------------------------------------------
def load_state_weather(state: str, start_dt, end_dt) -> pd.DataFrame:
    weather = pd.DataFrame(
        Weather.objects
        .filter(state=state, datetime__gte=start_dt, datetime__lte=end_dt)
        .order_by("datetime")
        .values("datetime", "temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms")
    )
    if weather.empty:
        raise ValueError(f"No weather rows for state={state} in range")

    weather = weather.rename(columns={"datetime": "ds"})
    weather["ds"] = pd.to_datetime(weather["ds"])
    return weather


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        df.sort_values("ds"),
        weather.sort_values("ds"),
        on="ds",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )


# ----------------------------------------------------------------------
# DAILY LOAD PROFILE  (mm-dd + 5-min slot -> typical load)
# ----------------------------------------------------------------------
def build_load_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Median load per (mm-dd, slot). Robust seasonal day-shape used to fill the
    `profile_y` feature and to synthesize the encoder window for far-future
    dates (mirrors the XGBoost predictor's profile logic)."""
    tmp = df.copy()
    tmp["slot"] = tmp["ds"].dt.hour * 12 + tmp["ds"].dt.minute // 5
    tmp["mmdd"] = tmp["ds"].dt.strftime("%m-%d")
    profile = (
        tmp.groupby(["mmdd", "slot"])["y"]
        .median()
        .reset_index()
        .rename(columns={"y": "profile_y"})
    )
    return profile


def attach_profile(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["slot"] = df["ds"].dt.hour * 12 + df["ds"].dt.minute // 5
    df["mmdd"] = df["ds"].dt.strftime("%m-%d")
    df = df.merge(profile, on=["mmdd", "slot"], how="left")
    # Feb-29 or any unseen (mmdd, slot): fall back to a slot-wide median.
    if df["profile_y"].isna().any():
        slot_med = profile.groupby("slot")["profile_y"].median()
        df["profile_y"] = df["profile_y"].fillna(df["slot"].map(slot_med))
    df["profile_y"] = df["profile_y"].interpolate().ffill().bfill()
    return df


# ----------------------------------------------------------------------
# WEATHER CLIMATOLOGY  (mm-dd + hour -> typical weather)
# Used at inference to fill the encoder window / any date live weather misses.
# ----------------------------------------------------------------------
def build_weather_climatology(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["mmdd"] = tmp["ds"].dt.strftime("%m-%d")
    tmp["hour"] = tmp["ds"].dt.hour
    clim = (
        tmp.groupby(["mmdd", "hour"])[
            ["temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms"]
        ]
        .mean()
        .reset_index()
    )
    return clim


# ----------------------------------------------------------------------
# SEASONAL LAGS
# ----------------------------------------------------------------------
def add_seasonal_lags(df: pd.DataFrame, source_col: str = "y") -> pd.DataFrame:
    """Add y_lag_24h (288 steps) and y_lag_7d (2016 steps).

    At training `source_col='y'` gives true shifted load. At inference the load
    is unknown, so the caller passes the profile column and we fill the head of
    the window with the profile itself — exactly what the XGBoost predictor does
    (`df['y_lag_24h'] = df['profile_y']`).
    """
    df = df.copy()
    df["y_lag_24h"] = df[source_col].shift(STEPS_PER_DAY)
    df["y_lag_7d"] = df[source_col].shift(STEPS_PER_DAY * 7)
    fill = df["profile_y"] if "profile_y" in df.columns else df[source_col]
    df["y_lag_24h"] = df["y_lag_24h"].fillna(fill)
    df["y_lag_7d"] = df["y_lag_7d"].fillna(fill)
    return df


# ----------------------------------------------------------------------
# FULL TRAINING FRAME
# ----------------------------------------------------------------------
def build_training_frame(state: str = "CG", max_days: int | None = None):
    """Build the model-ready training frame for `state`.

    Returns (df, profile, weather_climatology). `df` carries every column the
    TFT needs: time_idx, series, y and all KNOWN_REALS.

    `max_days` (optional) keeps only the most recent N days — used by the smoke
    test to validate the whole pipeline quickly.
    """
    df = load_state_target(state)

    if max_days is not None:
        cutoff = df["ds"].max() - pd.Timedelta(days=max_days)
        df = df[df["ds"] >= cutoff].reset_index(drop=True)

    start_dt, end_dt = df["ds"].min(), df["ds"].max()
    weather = load_state_weather(state, start_dt, end_dt)

    # Build climatology from the *raw merged* weather before any thinning.
    merged_for_clim = merge_weather(df[["ds"]].copy(), weather)
    weather_clim = build_weather_climatology(merged_for_clim)

    df = merge_weather(df, weather)
    df = add_calendar_features(df)      # is_weekend, is_holiday, season
    df = add_peak_features(df)          # hour, is_peak, *_x_hour

    profile = build_load_profile(df)
    df = attach_profile(df, profile)

    # 3-sigma outlier clamp on the target (matches XGBoost's clean_outliers).
    mu, sigma = df["y"].mean(), df["y"].std()
    df.loc[(df["y"] - mu).abs() > 3 * sigma, "y"] = np.nan
    df["y"] = df["y"].interpolate().bfill().ffill()

    df = add_seasonal_lags(df, source_col="y")

    # Any residual gaps in known reals -> safe fills.
    weather_cols = ["temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms"]
    df[weather_cols] = df[weather_cols].interpolate().ffill().bfill()
    df = df.dropna(subset=["y"]).reset_index(drop=True)

    # TFT bookkeeping columns.
    df["series"] = state
    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    return df, profile, weather_clim
