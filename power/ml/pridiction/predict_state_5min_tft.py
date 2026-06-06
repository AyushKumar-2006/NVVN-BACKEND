"""
Inference for the CG 5-min Temporal Fusion Transformer.

For a requested `forecast_date` we build a contiguous 5-min window made of:
  * an ENCODER tail  (MAX_ENCODER_LENGTH steps before the date) whose load is
    synthesized from the seasonal profile — the TFT analog of the XGBoost
    predictor feeding the profile in as the lag features, and
  * a DECODER day     (288 steps of the forecast date) whose known reals
    (weather, calendar, profile, seasonal lags) are fully specified.

The model then predicts the 288 decoder slots. This keeps the endpoint working
for any future date, even far beyond the training data.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from pytorch_forecasting import TimeSeriesDataSet

from power.utils.metadata import add_calendar_features
from power.ml.tft.config import GROUP_COL, STEPS_PER_DAY, TIME_IDX
from power.ml.tft.features_tft import add_peak_features, add_seasonal_lags, attach_profile
from power.ml.tft.train_tft import load_tft_artifact


# Cache the (model, artifact) so repeated requests don't reload from disk.
_CACHE: dict = {}


def _get_model(filename):
    if filename not in _CACHE:
        _CACHE[filename] = load_tft_artifact(filename)
    return _CACHE[filename]


def _weather_from_climatology(window: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Fill weather for every timestamp from the (mm-dd, hour) climatology."""
    df = window.copy()
    df["mmdd"] = df["ds"].dt.strftime("%m-%d")
    df["hour"] = df["ds"].dt.hour
    df = df.merge(clim, on=["mmdd", "hour"], how="left")

    wcols = ["temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms"]
    # Unseen mm-dd (e.g. Feb-29): fall back to hour-of-day means, then global.
    if df[wcols].isna().any().any():
        hour_means = clim.groupby("hour")[wcols].mean()
        for c in wcols:
            df[c] = df[c].fillna(df["hour"].map(hour_means[c]))
    df[wcols] = df[wcols].interpolate().ffill().bfill()
    return df.drop(columns=["mmdd", "hour"])


def _overlay_live_weather(df: pd.DataFrame, state: str, day) -> pd.DataFrame:
    """Best-effort: replace the decoder day's weather with a live forecast.
    Silently keeps climatology if the live fetch is unavailable (offline / far
    future date)."""
    try:
        from power.ml.trainy.common import merge_live_weather
        live = merge_live_weather(
            start_date=day, end_date=day, state=state, frequency="hourly"
        )
        live = live[["ds", "temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms"]].copy()
        live["ds"] = pd.to_datetime(live["ds"])
        merged = pd.merge_asof(
            df.sort_values("ds"),
            live.sort_values("ds"),
            on="ds", direction="nearest", tolerance=pd.Timedelta("1h"),
            suffixes=("", "_live"),
        )
        for c in ["temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms"]:
            lc = f"{c}_live"
            if lc in merged.columns:
                merged[c] = merged[lc].where(merged[lc].notna(), merged[c])
                merged = merged.drop(columns=[lc])
        return merged
    except Exception as exc:  # pragma: no cover - network/availability dependent
        print(f"[TFT] live weather unavailable ({exc}); using climatology.")
        return df


def predict_state_5min_tft(state: str, forecast_date, filename=None):
    """Return a DataFrame with columns ['ds', 'mw', 'temperature_c'] for the day."""
    from power.ml.tft.config import MODEL_FILENAME
    filename = filename or MODEL_FILENAME

    model, artifact = _get_model(filename)
    profile = artifact["profile"]
    clim = artifact["weather_climatology"]
    enc_len = artifact["max_encoder_length"]
    pred_len = artifact["max_prediction_length"]
    params = artifact["dataset_parameters"]

    forecast_date = pd.to_datetime(forecast_date).normalize()
    dec_start = forecast_date
    enc_start = dec_start - timedelta(minutes=5 * enc_len)

    # Contiguous encoder + decoder grid.
    window = pd.DataFrame({
        "ds": pd.date_range(
            enc_start,
            dec_start + timedelta(minutes=5 * (pred_len - 1)),
            freq="5min",
        )
    })

    # ---- weather ----
    window = _weather_from_climatology(window, clim)
    dec_mask = window["ds"] >= dec_start
    window = _overlay_live_weather(window, state, forecast_date.date())

    # ---- calendar + peak/interaction ----
    window = add_calendar_features(window)
    window = add_peak_features(window)

    # ---- daily profile + seasonal lags ----
    window = attach_profile(window, profile)
    window = add_seasonal_lags(window, source_col="profile_y")

    # ---- synthetic target history for the encoder ----
    # Decoder y is ignored by predict mode; encoder y = profile (the synthetic
    # recent load trajectory the TFT reads).
    window["y"] = window["profile_y"].astype(float)

    # ---- TFT bookkeeping ----
    window = window.sort_values("ds").reset_index(drop=True)
    # Use a group label the model was actually trained on (the CG series).
    trained_groups = list(params["categorical_encoders"][GROUP_COL].classes_)
    window[GROUP_COL] = state if state in trained_groups else trained_groups[0]
    window[TIME_IDX] = np.arange(len(window), dtype=np.int64)

    dataset = TimeSeriesDataSet.from_parameters(
        params, window, predict=True, stop_randomization=True,
    )

    raw = model.predict(
        dataset,
        mode="prediction",
        trainer_kwargs={"accelerator": "cpu", "logger": False, "enable_progress_bar": False},
    )
    preds = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
    preds = np.asarray(preds).reshape(-1)[:pred_len]

    decoder = window[window["ds"] >= dec_start].head(pred_len).reset_index(drop=True)
    out = pd.DataFrame({
        "ds": decoder["ds"],
        "mw": preds.astype(float),
        "temperature_c": decoder["temperature_c"].astype(float),
    })
    return out
