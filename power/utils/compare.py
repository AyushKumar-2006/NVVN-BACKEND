"""
Forecast-vs-actual comparison helpers (additive feature).

Powers two endpoints:
    GET /api/power/forecast-compare   - predicted vs actual for a single day
    GET /api/power/forecast-accuracy  - predicted vs actual over the last N days

Nothing here mutates the database or touches existing modules; it only reads
the trained 5-min model + stored load/weather data.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
from django.db.models import Avg, Count, Max, Sum
from ninja.errors import HttpError

from power.ml.model_store import load_model
from power.ml.pridiction.predict_state_5min import predict_state_5min_data
from power.ml.trainy.common import merge_live_weather
from power.models import StateLoad5Min, Weather
from power.utils.metadata import STATE_CODE_TO_NAME, add_calendar_features

DISCOMS = ["brpl", "bypl", "ndpl", "ndmc", "mes"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _energy_mu(sum_mw: float) -> float:
    """5-min MW samples -> energy in Million Units (MWh / 1000), matching the
    existing 5-min forecast response."""
    return round((sum_mw * (5 / 60)) / 1000, 2)


def _slot(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series)
    return dt.dt.hour * 12 + dt.dt.minute // 5


def _actual_daily(state: str, day: date):
    """Daily aggregates of actual load, or None when no data exists."""
    agg = (
        StateLoad5Min.objects
        .filter(state=state, datetime__date=day)
        .aggregate(avg=Avg("load_mw"), peak=Max("load_mw"),
                   total=Sum("load_mw"), n=Count("id"))
    )
    if not agg["n"] or agg["avg"] is None:
        return None
    return {
        "average_load_mw": round(agg["avg"], 2),
        "peak_load_mw": round(agg["peak"], 2),
        "energy_mu": _energy_mu(agg["total"]),
        "points": agg["n"],
    }


def _actual_5min(state: str, day: date) -> pd.DataFrame:
    df = pd.DataFrame(
        StateLoad5Min.objects
        .filter(state=state, datetime__date=day)
        .values("datetime", "load_mw")
    )
    if df.empty:
        return df
    df["slot"] = _slot(df["datetime"])
    return df[["slot", "load_mw"]]


def _diff(predicted: float, actual: float):
    mw = round(predicted - actual, 2)
    pct = round((mw / actual) * 100, 2) if actual else None
    return {"mw": mw, "pct": pct}


def _slot_accuracy(pred_df: pd.DataFrame, actual_df: pd.DataFrame):
    """Slot-aligned MAPE / accuracy between predicted and actual 5-min load."""
    pred = pred_df.copy()
    pred["slot"] = _slot(pred["ds"])
    merged = pred.merge(actual_df, on="slot", how="inner")
    merged = merged[merged["load_mw"] > 0]
    if merged.empty:
        return None, None, 0
    ape = (merged["mw"] - merged["load_mw"]).abs() / merged["load_mw"]
    mape = round(float(ape.mean()) * 100, 2)
    accuracy = round(max(0.0, 100 - mape), 2)
    return mape, accuracy, int(len(merged))


# ---------------------------------------------------------------------------
# fast batch predictor (weather from DB instead of a live call per day)
#   faithfully mirrors predict_state_5min_data's feature engineering
# ---------------------------------------------------------------------------
def _build_profile(state: str, day: pd.Timestamp) -> pd.DataFrame:
    hist = pd.DataFrame(
        StateLoad5Min.objects
        .filter(state=state, datetime__month=day.month, datetime__day=day.day)
        .values("datetime", "load_mw", *DISCOMS)
    )
    if hist.empty:
        return pd.DataFrame(columns=["slot", "profile_y"])

    hist["datetime"] = pd.to_datetime(hist["datetime"])
    hist["y"] = hist["load_mw"].fillna(hist[DISCOMS].sum(axis=1))

    q1, q3 = hist["y"].quantile(0.25), hist["y"].quantile(0.75)
    iqr = q3 - q1
    hist = hist[(hist["y"] >= q1 - 1.5 * iqr) & (hist["y"] <= q3 + 1.5 * iqr)]

    hist["slot"] = hist["datetime"].dt.hour * 12 + hist["datetime"].dt.minute // 5
    return (
        hist.groupby("slot")["y"].median()
        .reset_index().rename(columns={"y": "profile_y"})
    )


def _predict_one(model, state: str, day: date, weather_df: pd.DataFrame) -> pd.DataFrame:
    """One day of 5-min predictions using pre-supplied weather (DB-backed)."""
    fd = pd.to_datetime(day).normalize()

    df = pd.DataFrame({
        "ds": pd.date_range(fd, fd + timedelta(days=1) - timedelta(minutes=5), freq="5min")
    })
    df["slot"] = df["ds"].dt.hour * 12 + df["ds"].dt.minute // 5

    profile = _build_profile(state, fd)
    if profile.empty:
        raise ValueError(f"No historical data to build profile for {day}")

    df = df.merge(profile, on="slot", how="left")
    df["profile_y"] = df["profile_y"].interpolate().ffill().bfill()

    # weather: use DB rows if present, else fall back to a live fetch
    if weather_df is None or weather_df.empty:
        weather = merge_live_weather(fd.date(), fd.date(), state, "hourly")
    else:
        weather = weather_df.rename(columns={"datetime": "ds"}).copy()
        weather["ds"] = pd.to_datetime(weather["ds"])

    df = pd.merge_asof(
        df.sort_values("ds"), weather.sort_values("ds"),
        on="ds", direction="nearest", tolerance=pd.Timedelta("1h"),
    )

    df = add_calendar_features(df)
    df["hour"] = df["ds"].dt.hour
    df["is_peak"] = df["hour"].between(18, 23).astype(int)
    df["temp_x_hour"] = df["temperature_c"] * df["hour"]
    df["humidity_x_hour"] = df["humidity_pct"] * df["hour"]
    df["wind_x_hour"] = df["wind_speed_ms"] * df["hour"]
    for i in range(1, 7):
        df[f"y_lag_{i}"] = df["profile_y"]
    df["y_lag_24h"] = df["profile_y"]
    df["y_lag_7d"] = df["profile_y"]

    missing = set(model.feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

    df["mw"] = model.predict(df[model.feature_cols])
    return df[["ds", "mw", "temperature_c"]]


def _bulk_weather_by_date(state: str, days: list) -> dict:
    """One DB query -> {date: weather_df} for all requested days."""
    wdf = pd.DataFrame(
        Weather.objects
        .filter(state=state, datetime__date__in=days)
        .values("datetime", "temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms")
    )
    out = {}
    if wdf.empty:
        return out
    wdf["datetime"] = pd.to_datetime(wdf["datetime"])
    wdf["d"] = wdf["datetime"].dt.date
    for d, g in wdf.groupby("d"):
        out[d] = g.drop(columns=["d"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# endpoint builders
# ---------------------------------------------------------------------------
def build_forecast_compare(state: str, day) -> dict:
    try:
        if isinstance(day, str):
            day = date.fromisoformat(day.strip())
        elif not isinstance(day, date):
            raise ValueError("date must be a str or date")

        # Predicted: reuse the real 5-min forecast path (handles future dates
        # via climatology weather too).
        pred_df = predict_state_5min_data(state=state, forecast_date=day)
        if pred_df is None or pred_df.empty:
            raise ValueError("Empty prediction")

        pred_df["mw"] = pred_df["mw"].astype(float)
        predicted = {
            "average_load_mw": round(float(pred_df["mw"].mean()), 2),
            "peak_load_mw": round(float(pred_df["mw"].max()), 2),
            "energy_mu": _energy_mu(float(pred_df["mw"].sum())),
            "points": int(len(pred_df)),
        }

        actual = _actual_daily(state, day)

        result = {
            "state": STATE_CODE_TO_NAME.get(state, state),
            "state_code": state,
            "date": day.isoformat(),
            "actual_data_available": actual is not None,
            "predicted": predicted,
            "actual": actual,
            "difference": None,
            "mape_pct": None,
            "accuracy_score_pct": None,
            "points_compared": 0,
        }

        if actual is not None:
            actual_5min = _actual_5min(state, day)
            mape, accuracy, n = _slot_accuracy(pred_df, actual_5min)
            result["difference"] = {
                "average": _diff(predicted["average_load_mw"], actual["average_load_mw"]),
                "peak": _diff(predicted["peak_load_mw"], actual["peak_load_mw"]),
                "energy": _diff(predicted["energy_mu"], actual["energy_mu"]),
            }
            result["mape_pct"] = mape
            result["accuracy_score_pct"] = accuracy
            result["points_compared"] = n
        else:
            result["note"] = (
                "No actual load data stored for this date yet — showing the "
                "prediction only."
            )

        return result

    except HttpError:
        raise
    except Exception as e:
        raise HttpError(400, str(e))


def build_forecast_accuracy(state: str, days: int = 30) -> dict:
    try:
        days = max(1, min(int(days), 120))  # keep the report bounded

        latest = (
            StateLoad5Min.objects.filter(state=state)
            .order_by("-datetime")
            .values_list("datetime", flat=True)
            .first()
        )
        if latest is None:
            raise ValueError(f"No actual data for state={state}")

        latest_date = latest.date()
        start_date = latest_date - timedelta(days=days - 1)

        target_dates = list(
            StateLoad5Min.objects
            .filter(state=state, datetime__date__gte=start_date, datetime__date__lte=latest_date)
            .dates("datetime", "day")
        )
        if not target_dates:
            raise ValueError("No actual data in the requested window")

        model = load_model(f"state_5min_{state}.pkl")
        weather_by_date = _bulk_weather_by_date(state, target_dates)

        # bulk actual 5-min for the whole window (one query)
        actual_all = pd.DataFrame(
            StateLoad5Min.objects
            .filter(state=state, datetime__date__gte=start_date, datetime__date__lte=latest_date)
            .values("datetime", "load_mw")
        )
        actual_all["datetime"] = pd.to_datetime(actual_all["datetime"])
        actual_all["d"] = actual_all["datetime"].dt.date
        actual_all["slot"] = actual_all["datetime"].dt.hour * 12 + actual_all["datetime"].dt.minute // 5

        daily = []
        mapes = []
        for d in target_dates:
            try:
                pred_df = _predict_one(model, state, d, weather_by_date.get(d))
            except Exception as e:
                daily.append({"date": d.isoformat(), "error": str(e)})
                continue

            a = actual_all[actual_all["d"] == d][["slot", "load_mw"]]
            mape, accuracy, n = _slot_accuracy(pred_df, a)

            pred_avg = round(float(pred_df["mw"].mean()), 2)
            pred_peak = round(float(pred_df["mw"].max()), 2)
            act_avg = round(float(a["load_mw"].mean()), 2)
            act_peak = round(float(a["load_mw"].max()), 2)

            if mape is not None:
                mapes.append(mape)

            daily.append({
                "date": d.isoformat(),
                "predicted_avg_mw": pred_avg,
                "actual_avg_mw": act_avg,
                "predicted_peak_mw": pred_peak,
                "actual_peak_mw": act_peak,
                "error_pct": mape,
                "accuracy_pct": accuracy,
                "points_compared": n,
            })

        overall_mape = round(float(np.mean(mapes)), 2) if mapes else None
        overall_accuracy = round(max(0.0, 100 - overall_mape), 2) if overall_mape is not None else None

        # best / worst day by accuracy (only over days actually evaluated)
        evaluated = [r for r in daily if r.get("accuracy_pct") is not None]
        best = max(evaluated, key=lambda r: r["accuracy_pct"]) if evaluated else None
        worst = min(evaluated, key=lambda r: r["accuracy_pct"]) if evaluated else None

        def _extreme(r):
            if not r:
                return None
            return {
                "date": r["date"],
                "accuracy_pct": r["accuracy_pct"],
                "error_pct": r["error_pct"],
                "predicted_avg_mw": r["predicted_avg_mw"],
                "actual_avg_mw": r["actual_avg_mw"],
            }

        return {
            "state": STATE_CODE_TO_NAME.get(state, state),
            "state_code": state,
            "days_requested": days,
            "days_evaluated": len(mapes),
            "date_range": {
                "from": target_dates[0].isoformat(),
                "to": target_dates[-1].isoformat(),
            },
            "overall": {
                "mape_pct": overall_mape,
                "accuracy_pct": overall_accuracy,
            },
            "best_day": _extreme(best),
            "worst_day": _extreme(worst),
            "daily": daily,
        }

    except HttpError:
        raise
    except Exception as e:
        raise HttpError(400, str(e))
