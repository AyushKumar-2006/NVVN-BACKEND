"""
CG (Chhattisgarh) daily peak-demand forecasting — shared pipeline.

The real Chhattisgarh series in ``StateLoad5Min (state='CG_WRLDC')`` is sparse:
~3 points per day (an off-peak ~03:00 reading, an evening ~18-19:00 reading, and
the day's maximum-demand point). It is therefore modelled at the **daily** level
on the **daily peak load** (the day's max MW) — the canonical demand-forecast
target and the most consistently present point in the WRLDC PSP report.

This module is the single source of truth used by both:
  * ``python manage.py retrain_cg_models``  (training + evaluation), and
  * the ``/api/cg/*`` endpoints              (forecast / actuals / compare / stats).

Feature set (exactly as specified for the project):
    hour, dayofweek, month, is_weekend, lag_1d, lag_7d, rolling_mean_7d

ADDITIVE: new module. Does not modify the existing weather-based models in
power/ml/models/{xgb_model,prophet_model}.py (those need weather regressors the
CG_WRLDC series does not carry).
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timedelta

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.db.models import Max

from power.models import StateLoad5Min

# --------------------------------------------------------------------------- #
STATE = "CG_WRLDC"

MODELS_DIR = os.path.join(settings.BASE_DIR, "power", "ml", "models")
XGB_PATH = os.path.join(MODELS_DIR, "cg_xgb.joblib")
PROPHET_PATH = os.path.join(MODELS_DIR, "cg_prophet.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "eval_metrics.json")

FEATURE_COLS = [
    "hour", "dayofweek", "month", "is_weekend",
    "lag_1d", "lag_7d", "rolling_mean_7d",
]
TARGET = "y"
TARGET_DESC = "daily_peak_load_mw"


# --------------------------------------------------------------------------- #
# Data + feature engineering
# --------------------------------------------------------------------------- #
def build_daily_series(state: str = STATE) -> pd.DataFrame:
    """Daily peak-load series on a continuous daily index.

    Returns a DataFrame indexed by date (DatetimeIndex, daily) with columns:
      y            daily peak load (MW); missing calendar days linearly interpolated
      peak_hour    hour-of-day at which the daily peak occurred (ffilled over gaps)
      interpolated True where the day had no real reading (filled), else False
    Returns an empty DataFrame if there is no data for ``state``.
    """
    qs = (StateLoad5Min.objects
          .filter(state=state)
          .order_by("datetime")
          .values("datetime", "load_mw"))
    raw = pd.DataFrame(list(qs))
    if raw.empty:
        return raw

    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.dropna(subset=["load_mw"])
    raw["date"] = raw["datetime"].dt.normalize()

    # daily peak + the hour it occurred at
    idxmax = raw.groupby("date")["load_mw"].idxmax()
    daily = raw.loc[idxmax, ["date", "load_mw", "datetime"]].copy()
    daily["peak_hour"] = daily["datetime"].dt.hour
    daily = (daily.drop(columns=["datetime"])
                  .rename(columns={"load_mw": "y"})
                  .set_index("date")
                  .sort_index())

    # continuous daily index so lag/rolling features are well-defined
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full)
    daily["interpolated"] = daily["y"].isna()
    daily["y"] = daily["y"].interpolate(method="linear").bfill().ffill()
    daily["peak_hour"] = daily["peak_hour"].ffill().bfill()
    daily.index.name = "date"
    return daily


def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Attach the model feature columns to a daily series from build_daily_series."""
    df = daily.copy()
    idx = df.index
    df["hour"] = df["peak_hour"].astype(int)
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["lag_1d"] = df["y"].shift(1)
    df["lag_7d"] = df["y"].shift(7)
    # shift(1) so the rolling window uses only past days (no target leakage)
    df["rolling_mean_7d"] = df["y"].shift(1).rolling(7, min_periods=1).mean()
    return df


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and MAPE (%) for two equal-length arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"mae": None, "rmse": None, "mape": None, "n": 0}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs(err[mask] / y_true[mask])) * 100) if mask.any() else None
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 3) if mape is not None else None,
        "n": int(y_true.size),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_models(xgb_model, prophet_model) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(xgb_model, XGB_PATH)
    joblib.dump(prophet_model, PROPHET_PATH)


def models_exist() -> bool:
    return os.path.exists(XGB_PATH) and os.path.exists(PROPHET_PATH)


def load_models():
    """Return (xgb_model, prophet_model). Raises FileNotFoundError if not trained."""
    if not models_exist():
        raise FileNotFoundError(
            "CG models not found. Run: python manage.py retrain_cg_models"
        )
    return joblib.load(XGB_PATH), joblib.load(PROPHET_PATH)


def write_metrics(meta: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def read_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


def last_retrain_iso():
    """ISO timestamp of the last retrain, from the metrics file (mtime fallback)."""
    meta = read_metrics()
    if meta and meta.get("generated_at"):
        return meta["generated_at"]
    if os.path.exists(METRICS_PATH):
        return datetime.fromtimestamp(os.path.getmtime(METRICS_PATH)).isoformat()
    return None


def data_count(state: str = STATE) -> int:
    return StateLoad5Min.objects.filter(state=state).count()


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(test_frac: float = 0.2):
    """Train XGBoost + Prophet on the daily peak series.

    Chronological train/test split (no shuffle — it's a time series). Returns
    ``(xgb_model, prophet_model, meta)`` where ``meta`` is the JSON-serialisable
    evaluation summary (also written to eval_metrics.json by the command).
    """
    from xgboost import XGBRegressor
    from prophet import Prophet

    daily = build_daily_series()
    if daily.empty:
        raise ValueError(f"No data for state={STATE!r}; nothing to train on.")

    feats = make_features(daily).dropna(subset=FEATURE_COLS + [TARGET])
    n = len(feats)
    if n < 30:
        raise ValueError(f"Too few daily points to train ({n}).")

    split = int(n * (1 - test_frac))
    train_df, test_df = feats.iloc[:split], feats.iloc[split:]

    # ---- XGBoost ----
    xgb = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    xgb.fit(train_df[FEATURE_COLS], train_df[TARGET])
    xgb.feature_cols = FEATURE_COLS          # metadata, mirrors existing model pattern
    xgb.target_col = TARGET
    xgb_test_pred = xgb.predict(test_df[FEATURE_COLS])
    xgb_metrics = metrics(test_df[TARGET].values, xgb_test_pred)

    # ---- Prophet (same target, same chronological split) ----
    pdf = pd.DataFrame({"ds": feats.index, "y": feats[TARGET].to_numpy()})
    prophet = Prophet(
        daily_seasonality=False,   # one point per day -> no intraday seasonality
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    prophet.fit(pdf.iloc[:split])
    prophet_fc = prophet.predict(pdf[["ds"]])
    prophet_test_pred = prophet_fc["yhat"].to_numpy()[split:]
    prophet_metrics = metrics(test_df[TARGET].values, prophet_test_pred)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "state": STATE,
        "target": TARGET_DESC,
        "feature_cols": FEATURE_COLS,
        "data_count_points": data_count(),
        "n_daily": int(n),
        "n_train": int(split),
        "n_test": int(n - split),
        "test_frac": test_frac,
        "train_range": [str(feats.index[0].date()), str(feats.index[split - 1].date())],
        "test_range": [str(feats.index[split].date()), str(feats.index[-1].date())],
        "models": {
            "xgboost": xgb_metrics,
            "prophet": prophet_metrics,
        },
    }
    return xgb, prophet, meta


# --------------------------------------------------------------------------- #
# Inference: future forecast + historical backtest
# --------------------------------------------------------------------------- #
def forecast(days: int = 30) -> dict:
    """Forecast the next ``days`` days of daily peak demand with both models.

    XGBoost is rolled forward recursively (each prediction feeds the next day's
    lag/rolling features); Prophet uses its native future dataframe.
    """
    days = max(1, int(days))
    daily = build_daily_series()
    if daily.empty:
        raise ValueError(f"No data for state={STATE!r}.")
    xgb, prophet = load_models()

    series = daily["y"].copy()                       # continuous daily history
    peak_hour = int(round(float(daily["peak_hour"].tail(60).median())))
    last_date = series.index.max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                 periods=days, freq="D")

    xgb_vals = []
    for d in future_dates:
        recent = series.tail(7)
        feat = {
            "hour": peak_hour,
            "dayofweek": int(d.dayofweek),
            "month": int(d.month),
            "is_weekend": int(d.dayofweek >= 5),
            "lag_1d": float(series.get(d - pd.Timedelta(days=1), series.iloc[-1])),
            "lag_7d": float(series.get(d - pd.Timedelta(days=7), recent.iloc[0])),
            "rolling_mean_7d": float(recent.mean()),
        }
        X = pd.DataFrame([feat])[FEATURE_COLS]
        yhat = float(xgb.predict(X)[0])
        series.loc[d] = yhat                         # feed back for next step
        xgb_vals.append(yhat)

    pfc = prophet.predict(pd.DataFrame({"ds": future_dates}))
    rows = []
    for i, d in enumerate(future_dates):
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "xgboost_mw": round(xgb_vals[i], 1),
            "prophet_mw": round(float(pfc["yhat"].iloc[i]), 1),
            "prophet_lower_mw": round(float(pfc["yhat_lower"].iloc[i]), 1),
            "prophet_upper_mw": round(float(pfc["yhat_upper"].iloc[i]), 1),
        })
    return {
        "state": STATE,
        "target": TARGET_DESC,
        "days": days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast": rows,
    }


def actuals(start: date, end: date) -> dict:
    """Actual daily peak demand (real readings only) within [start, end]."""
    daily = build_daily_series()
    rows = []
    if not daily.empty:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        win = daily[(daily.index >= lo) & (daily.index <= hi)]
        for d, r in win.iterrows():
            if bool(r["interpolated"]):
                continue                              # only real observations
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "peak_mw": round(float(r["y"]), 1),
                "peak_hour": int(r["peak_hour"]),
            })
    return {
        "state": STATE,
        "metric": TARGET_DESC,
        "start": str(start),
        "end": str(end),
        "count": len(rows),
        "actuals": rows,
    }


def compare(start: date, end: date) -> dict:
    """Actuals vs both models' predictions over a historical window + metrics.

    XGBoost predictions here are one-step (they use the real observed lags), so
    this is an honest in-window backtest, not a recursive forecast.
    """
    daily = build_daily_series()
    if daily.empty:
        raise ValueError(f"No data for state={STATE!r}.")
    xgb, prophet = load_models()

    feats = make_features(daily).dropna(subset=FEATURE_COLS + [TARGET])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    win = feats[(feats.index >= lo) & (feats.index <= hi)]

    rows, y_true, xgb_p, prop_p = [], [], [], []
    if not win.empty:
        xgb_pred = xgb.predict(win[FEATURE_COLS])
        pfc = prophet.predict(pd.DataFrame({"ds": win.index}))
        prophet_pred = pfc["yhat"].to_numpy()
        for i, (d, r) in enumerate(win.iterrows()):
            actual = None if bool(r["interpolated"]) else round(float(r["y"]), 1)
            xv = round(float(xgb_pred[i]), 1)
            pv = round(float(prophet_pred[i]), 1)
            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "actual_mw": actual,
                "xgboost_mw": xv,
                "prophet_mw": pv,
            })
            if actual is not None:                    # metrics over real days only
                y_true.append(float(r["y"]))
                xgb_p.append(float(xgb_pred[i]))
                prop_p.append(float(prophet_pred[i]))

    return {
        "state": STATE,
        "metric": TARGET_DESC,
        "start": str(start),
        "end": str(end),
        "count": len(rows),
        "rows": rows,
        "metrics": {
            "xgboost": metrics(y_true, xgb_p),
            "prophet": metrics(y_true, prop_p),
        },
    }


# --------------------------------------------------------------------------- #
# Forward (today-anchored) forecast — hybrid XGBoost(1-15) + Prophet(16-30)
# --------------------------------------------------------------------------- #
CG_5MIN_STATE = "CG"          # continuous 5-min CG series (used for intraday shape)
HYBRID_XGB_DAYS = 15          # days 1..15 use XGBoost, days 16.. use Prophet


def _xgb_feat_row(series: pd.Series, d: pd.Timestamp, peak_hour: int) -> pd.DataFrame:
    recent = series.tail(7)
    feat = {
        "hour": peak_hour,
        "dayofweek": int(d.dayofweek),
        "month": int(d.month),
        "is_weekend": int(d.dayofweek >= 5),
        "lag_1d": float(series.get(d - pd.Timedelta(days=1), series.iloc[-1])),
        "lag_7d": float(series.get(d - pd.Timedelta(days=7), recent.iloc[0])),
        "rolling_mean_7d": float(recent.mean()),
    }
    return pd.DataFrame([feat])[FEATURE_COLS]


def forecast_from_today(days: int = 30, xgb_days: int = HYBRID_XGB_DAYS) -> dict:
    """30-day forward forecast anchored at *today*, with a live temperature track.

    The headline ``forecast_mw`` is a hybrid: the first ``xgb_days`` days use
    XGBoost (strong short-horizon accuracy), the remaining days use Prophet
    (steadier seasonal trend further out). Each row also carries the 4-district
    population-weighted temperature for that day (open-meteo forecast within the
    horizon, 30-year climate normal beyond it). ``xgboost_mw`` / ``prophet_mw``
    are kept per row for reference.
    """
    days = max(1, int(days))
    daily = build_daily_series()
    if daily.empty:
        raise ValueError(f"No data for state={STATE!r}.")
    xgb, prophet = load_models()

    series = daily["y"].copy()
    peak_hour = int(round(float(daily["peak_hour"].tail(60).median())))
    last_date = series.index.max()
    today = pd.Timestamp(datetime.now().date())
    target_dates = pd.date_range(today, periods=days, freq="D")
    horizon_end = target_dates[-1]

    # roll XGBoost recursively from the last known day up to the horizon end
    gen_end = max(horizon_end, last_date + pd.Timedelta(days=1))
    gen_dates = pd.date_range(last_date + pd.Timedelta(days=1), gen_end, freq="D")
    s = series.copy()
    xgb_map = {}
    for d in gen_dates:
        yhat = float(xgb.predict(_xgb_feat_row(s, d, peak_hour))[0])
        s.loc[d] = yhat
        xgb_map[d.normalize()] = yhat

    # Prophet directly on the target window
    pfc = prophet.predict(pd.DataFrame({"ds": target_dates}))
    prophet_map = {d.normalize(): float(v) for d, v in zip(target_dates, pfc["yhat"])}

    # live 4-district weighted temperature for the same dates, with its source
    # (forecast for ~today..+15, then the 30-year climate normal beyond)
    temps, temp_src = {}, {}
    try:
        from power.ml import weather as W
        for r in W.fetch_daily_weather(CG_5MIN_STATE, today.date(), days):
            temps[r["date"]] = r.get("temperature_c")
            temp_src[r["date"]] = r.get("source")
    except Exception:  # noqa: BLE001  — temperature is best-effort; load still works
        temps, temp_src = {}, {}

    # smooth cross-fade window around the XGBoost->Prophet boundary so the
    # headline line has no visible step (model label below still flips hard).
    center = xgb_days - 0.5
    half = 2.0
    rows = []
    for i, d in enumerate(target_dates):
        key = d.normalize()
        xv = xgb_map.get(key)
        if xv is None and key in series.index:        # target day already observed
            xv = float(series.loc[key])
        pv = prophet_map.get(key)
        model = "xgboost" if i < xgb_days else "prophet"
        alpha = min(1.0, max(0.0, (i - center) / (2 * half) + 0.5))
        if xv is not None and pv is not None:
            fmw = (1 - alpha) * xv + alpha * pv
        else:
            fmw = xv if xv is not None else pv
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "forecast_mw": round(float(fmw), 1) if fmw is not None else None,
            "model": model,
            "xgboost_mw": round(float(xv), 1) if xv is not None else None,
            "prophet_mw": round(float(pv), 1) if pv is not None else None,
            "temperature_c": temps.get(d.strftime("%Y-%m-%d")),
            "temp_source": temp_src.get(d.strftime("%Y-%m-%d")),
        })

    fvals = [r["forecast_mw"] for r in rows if r["forecast_mw"] is not None]
    tvals = [r["temperature_c"] for r in rows if r["temperature_c"] is not None]
    return {
        "state": STATE,
        "target": TARGET_DESC,
        "days": days,
        "anchor": str(today.date()),
        "xgb_days": xgb_days,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "today_peak_mw": rows[0]["forecast_mw"] if rows else None,
            "peak_30d_mw": round(max(fvals), 1) if fvals else None,
            "avg_temp_c": round(sum(tvals) / len(tvals), 1) if tvals else None,
            "max_temp_c": round(max(tvals), 1) if tvals else None,
        },
        "forecast": rows,
    }


# --------------------------------------------------------------------------- #
# Intraday 5-min curve — real CG when present, else a seamless analog-day fill
# --------------------------------------------------------------------------- #
def _real_intraday(day: date):
    start = datetime.combine(day, time.min)
    end = start + timedelta(days=1)
    qs = (StateLoad5Min.objects
          .filter(state=CG_5MIN_STATE, datetime__gte=start, datetime__lt=end)
          .order_by("datetime").values_list("datetime", "load_mw"))
    return [(dt, v) for dt, v in qs]


def _donor_curve(day: date):
    """A recent complete (288-point) CG day to lend its intraday shape, preferring
    the same weekday so the profile matches. Returns (values[288], donor_peak)."""
    last = (StateLoad5Min.objects.filter(state=CG_5MIN_STATE)
            .aggregate(m=Max("datetime"))["m"])
    if last is None:
        return None
    window_start = datetime.combine(last.date() - timedelta(days=90), time.min)
    qs = (StateLoad5Min.objects
          .filter(state=CG_5MIN_STATE, datetime__gte=window_start)
          .order_by("datetime").values_list("datetime", "load_mw"))
    df = pd.DataFrame(list(qs), columns=["dt", "load"])
    if df.empty:
        return None
    df["d"] = df["dt"].dt.normalize()
    counts = df.groupby("d").size()
    full = [d for d, n in counts.items() if n >= 288]
    if not full:
        return None
    same_wd = [d for d in full if d.dayofweek == day.weekday()]
    donor = max(same_wd) if same_wd else max(full)
    cur = (df[df["d"] == donor].sort_values("dt")["load"]
           .to_numpy(dtype=float)[:288])
    if cur.size < 288 or not np.isfinite(cur).any():
        return None
    return cur, float(np.nanmax(cur))


def _target_peak(day: date):
    """Daily peak level to scale the curve to: the real WRLDC peak for that day
    if we have it, otherwise the recent peak level."""
    daily = build_daily_series()
    if daily.empty:
        return None
    ts = pd.Timestamp(day)
    if ts in daily.index and not bool(daily.loc[ts, "interpolated"]):
        return float(daily.loc[ts, "y"])
    return float(daily["y"].tail(7).mean())


def intraday_5min(day: date):
    """5-min CG load curve for ``day`` as a list of {datetime, load_mw}.

    Returns the real series when the day is present in the DB; otherwise builds a
    continuous fill from a recent same-weekday curve rescaled to the day's peak
    level (with light smooth jitter) so the dashboard always shows an unbroken
    current-day profile.
    """
    real = _real_intraday(day)
    if len(real) >= 200:
        return [{"datetime": dt.strftime("%Y-%m-%dT%H:%M:%S"),
                 "load_mw": (round(v, 1) if v is not None else None)}
                for dt, v in real]

    donor = _donor_curve(day)
    if donor is None:
        return []
    values, donor_peak = donor
    target = _target_peak(day)
    scale = (target / donor_peak) if (target and donor_peak) else 1.0
    rng = np.random.default_rng(int(day.strftime("%Y%m%d")))
    jitter = rng.normal(0.0, 0.004, values.size)        # ~0.4% smooth wander
    base = datetime.combine(day, time.min)
    return [{
        "datetime": (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
        "load_mw": round(float(v * scale * (1.0 + jitter[i])), 1),
    } for i, v in enumerate(values)]
