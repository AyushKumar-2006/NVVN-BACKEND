from datetime import date, datetime, timedelta
import time
import pandas as pd
import requests
from power.utils.logger import get_logger

logger = get_logger("WeatherFetch")





STATE_COORDS = {
    "DL": {"lat": 28.6139, "lon": 77.2090},
    "MH": {"lat": 19.7515, "lon": 75.7139},
    "TN": {"lat": 11.1271, "lon": 78.6569},
    "UP": {"lat": 26.8467, "lon": 80.9462},
    "AP": {"lat": 15.9129, "lon": 79.7400},
    "AR": {"lat": 28.2180, "lon": 94.7278},
    "AS": {"lat": 26.2006, "lon": 92.9376},
    "BR": {"lat": 25.0961, "lon": 85.3131},
    "CH": {"lat": 30.7333, "lon": 76.7794},
    "CG": {"lat": 21.2787, "lon": 81.8661},
    "GA": {"lat": 15.2993, "lon": 74.1240},
    "GJ": {"lat": 22.2587, "lon": 71.1924},
    "HR": {"lat": 29.0588, "lon": 76.0856},
    "HP": {"lat": 31.1048, "lon": 77.1734},
    "JK": {"lat": 33.7782, "lon": 76.5762},
    "JH": {"lat": 23.6102, "lon": 85.2799},
    "KA": {"lat": 15.3173, "lon": 75.7139},
    "KL": {"lat": 10.8505, "lon": 76.2711},
    "MN": {"lat": 24.6637, "lon": 93.9063},
    "ML": {"lat": 25.4670, "lon": 91.3662},
    "MZ": {"lat": 23.1645, "lon": 92.9376},
    "MP": {"lat": 22.9734, "lon": 78.6569},
    "NL": {"lat": 26.1584, "lon": 94.5624},
    "OD": {"lat": 20.9517, "lon": 85.0985},
    "PY": {"lat": 11.9416, "lon": 79.8083},
    "PB": {"lat": 31.1471, "lon": 75.3412},
    "RJ": {"lat": 27.0238, "lon": 74.2179},
    "SK": {"lat": 27.5330, "lon": 88.5122},
    "TS": {"lat": 18.1124, "lon": 79.0193},
    "TR": {"lat": 23.9408, "lon": 91.9882},
    "UK": {"lat": 30.0668, "lon": 79.0193},
    "WB": {"lat": 22.9868, "lon": 87.8550},
}


# ---------------------------------------------------------------------------
# Original single-day reference draft (kept for reference; superseded by the
# bulk-range + climatology implementation below).
# ---------------------------------------------------------------------------
# def fetch_weather(state_short: str, date_str: str, frequency="hourly") -> pd.DataFrame:
#     if state_short not in STATE_COORDS:
#         raise ValueError(f"State coords missing: {state_short}")

#     coords = STATE_COORDS[state_short]
#     target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
#     today = datetime.now().date()

#     if target_date < today:
#         url = "https://archive-api.open-meteo.com/v1/archive"
#         api_type = "ARCHIVE"
#     else:
#         url = "https://api.open-meteo.com/v1/forecast"
#         api_type = "FORECAST"

#     params = {
#         "latitude": round(coords["lat"], 4),
#         "longitude": round(coords["lon"], 4),
#         "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation",
#         "start_date": date_str,
#         "end_date": date_str,
#         "timezone": "Asia/Kolkata",
#     }

#     try:
#         r = requests.get(url, params=params, timeout=20)
#         r.raise_for_status()
#         data = r.json()
#         logger.info(f"[{state_short}] {api_type} weather fetched for {date_str}")
#     except Exception as e:
#         logger.error(f"[{state_short}] Weather fetch failed: {e}")
#         return pd.DataFrame()


#     if "hourly" not in data:
#         return pd.DataFrame()

#     df = pd.DataFrame({
#         "datetime": data["hourly"]["time"],
#         "temperature_c": data["hourly"]["temperature_2m"],
#         "humidity_pct": data["hourly"]["relativehumidity_2m"],
#         "wind_speed_ms": data["hourly"]["windspeed_10m"],
#         "rain_mm": data["hourly"]["precipitation"],
#     })

#     weather = weather.set_index("ds").resample("5min").interpolate("time").reset_index()
#     weather["state"] = weather["state"].ffill()
#     weather["frequency"] = weather["frequency"].ffill()

#     print(df.head())

#     return df


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _to_date(value):
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _request_open_meteo(coords, start_str, end_str, url, state_short, frequency):
    """Single open-meteo request for a whole [start, end] range -> 5-min df."""
    params = {
        "latitude": round(coords["lat"], 4),
        "longitude": round(coords["lon"], 4),
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation",
        "start_date": start_str,
        "end_date": end_str,
        "timezone": "Asia/Kolkata",
    }

    data = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            logger.info(f"[{state_short}] weather fetched {start_str}..{end_str}")
            break
        except Exception as e:
            if attempt == 2:
                logger.error(
                    f"[{state_short}] Weather fetch failed {start_str}..{end_str}: {e}"
                )
                return pd.DataFrame()
            time.sleep(2)

    if not data or "hourly" not in data or not data["hourly"].get("time"):
        return pd.DataFrame()

    df = pd.DataFrame({
        "datetime": data["hourly"]["time"],
        "temperature_c": data["hourly"]["temperature_2m"],
        "humidity_pct": data["hourly"]["relativehumidity_2m"],
        "wind_speed_ms": data["hourly"]["windspeed_10m"],
        "rain_mm": data["hourly"]["precipitation"],
    })

    dt = pd.to_datetime(df["datetime"])
    if dt.dt.tz is not None:
        dt = dt.dt.tz_localize(None)
    df["datetime"] = dt

    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    # 5-MIN INTERPOLATION
    df = (
        df.set_index("datetime")
        .resample("5min")
        .interpolate("time")
        .reset_index()
    )

    df["state"] = state_short
    df["frequency"] = frequency
    return df


def _fetch_climatology(coords, start_dt, end_dt, state_short, frequency, today):
    """
    Future dates beyond the forecast horizon have no real weather available.
    Use the same calendar window from the most recent prior year that the
    archive can serve, then shift the timestamps forward to the target year.
    """
    for years_back in range(1, 6):
        try:
            proxy_start = start_dt.replace(year=start_dt.year - years_back)
            proxy_end = end_dt.replace(year=end_dt.year - years_back)
        except ValueError:  # e.g. Feb 29 -> fall back one day
            proxy_start = (start_dt - timedelta(days=1)).replace(year=start_dt.year - years_back)
            proxy_end = (end_dt - timedelta(days=1)).replace(year=end_dt.year - years_back)

        if proxy_end >= today:
            continue

        df = _request_open_meteo(
            coords, proxy_start.isoformat(), proxy_end.isoformat(),
            ARCHIVE_URL, state_short, frequency,
        )
        if not df.empty:
            df["datetime"] = df["datetime"] + pd.DateOffset(years=years_back)
            logger.warning(
                f"[{state_short}] Forecast horizon exceeded; using {proxy_start.year} "
                f"climatology proxy for {start_dt}..{end_dt}"
            )
            return df

    return pd.DataFrame()


def fetch_weather(state_short: str, date_str: str, frequency="hourly") -> pd.DataFrame:
    return fetch_weather_range(state_short, date_str, date_str, frequency)


def fetch_weather_range(state_short: str, start_date, end_date, frequency="hourly") -> pd.DataFrame:
    if state_short not in STATE_COORDS:
        raise ValueError(f"State coords missing: {state_short}")

    coords = STATE_COORDS[state_short]
    start_dt = _to_date(start_date)
    end_dt = _to_date(end_date)
    today = datetime.now().date()

    frames = []

    # ---- PAST -> ARCHIVE (single bulk request for the whole span) ----
    archive_end = min(end_dt, today - timedelta(days=1))
    if start_dt <= archive_end:
        df = _request_open_meteo(
            coords, start_dt.isoformat(), archive_end.isoformat(),
            ARCHIVE_URL, state_short, frequency,
        )
        if not df.empty:
            frames.append(df)
        else:
            logger.warning(f"[{state_short}] No archive weather {start_dt}..{archive_end}")

    # ---- TODAY / FUTURE -> FORECAST (single request, climatology fallback) ----
    if end_dt >= today:
        fc_start = max(start_dt, today)
        df = _request_open_meteo(
            coords, fc_start.isoformat(), end_dt.isoformat(),
            FORECAST_URL, state_short, frequency,
        )
        if df.empty:
            df = _fetch_climatology(coords, fc_start, end_dt, state_short, frequency, today)
        if not df.empty:
            frames.append(df)
        else:
            logger.warning(f"[{state_short}] No forecast weather {fc_start}..{end_dt}")

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


# ===========================================================================
# DAILY WEATHER (used by the 30-day forecast)
#   - Open-Meteo forecast API covers ~16 days from today (free)
#   - Beyond that, fall back to climatology: average of the same calendar day
#     across the past N years from the free archive API
# ===========================================================================

# Open-Meteo's free forecast serves today..today+15 (16 days inclusive);
# requesting an end_date beyond that fails the whole call, so cap here and let
# climatology cover the remaining days.
FORECAST_HORIZON_DAYS = 15


def _request_daily(coords, start_dt, end_dt, url, state_short):
    """Single open-meteo request -> daily mean/max temperature DataFrame."""
    params = {
        "latitude": round(coords["lat"], 4),
        "longitude": round(coords["lon"], 4),
        "daily": "temperature_2m_mean,temperature_2m_max",
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "timezone": "Asia/Kolkata",
    }

    data = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                logger.error(
                    f"[{state_short}] Daily weather failed {start_dt}..{end_dt}: {e}"
                )
                return pd.DataFrame()
            time.sleep(2)

    daily = (data or {}).get("daily", {})
    times = daily.get("time") or []
    if not times:
        return pd.DataFrame()

    return pd.DataFrame({
        "date": [datetime.strptime(t, "%Y-%m-%d").date() for t in times],
        "temp": daily.get("temperature_2m_mean"),
        "temp_max": daily.get("temperature_2m_max"),
    })


def _climatology_daily(coords, target_dates, state_short, years, today):
    """
    For future dates beyond the forecast horizon: average the daily mean
    temperature of the same calendar day (month, day) across the past `years`
    years pulled from the free archive API.
    """
    start, end = min(target_dates), max(target_dates)
    by_md = {}  # (month, day) -> [temps]

    for yb in range(1, years + 1):
        try:
            ps = start.replace(year=start.year - yb)
            pe = end.replace(year=end.year - yb)
        except ValueError:  # Feb 29 -> shift one day
            ps = (start - timedelta(days=1)).replace(year=start.year - yb)
            pe = (end - timedelta(days=1)).replace(year=end.year - yb)

        if pe >= today:  # archive cannot serve very recent dates
            continue

        df = _request_daily(coords, ps, pe, ARCHIVE_URL, state_short)
        for _, row in df.iterrows():
            if row["temp"] is None:
                continue
            md = (row["date"].month, row["date"].day)
            by_md.setdefault(md, []).append(float(row["temp"]))

    out = {}
    for d in target_dates:
        vals = by_md.get((d.month, d.day), [])
        out[d] = round(sum(vals) / len(vals), 2) if vals else None
    return out


def fetch_daily_weather(state_short: str, from_date, days: int = 30, climatology_years: int = 3):
    """
    Daily mean temperature for `days` days starting at `from_date`.

    Returns a list aligned to the date range. Each item is a dict
    {"date", "temperature_c", "source"} where source is
    "forecast" | "archive" | "climatology".
    """
    if state_short not in STATE_COORDS:
        raise ValueError(f"State coords missing: {state_short}")

    coords = STATE_COORDS[state_short]
    if isinstance(from_date, str):
        from_date = _to_date(from_date)

    today = datetime.now().date()
    horizon = today + timedelta(days=FORECAST_HORIZON_DAYS)
    target_dates = [from_date + timedelta(days=i) for i in range(days)]

    temps = {}   # date -> temp
    source = {}  # date -> source label

    # 1) past dates -> archive
    past = [d for d in target_dates if d < today]
    if past:
        df = _request_daily(coords, min(past), max(past), ARCHIVE_URL, state_short)
        for _, row in df.iterrows():
            temps[row["date"]] = (None if row["temp"] is None else round(float(row["temp"]), 2))
            source[row["date"]] = "archive"

    # 2) within forecast horizon -> forecast API
    fc = [d for d in target_dates if today <= d <= horizon]
    if fc:
        df = _request_daily(coords, min(fc), max(fc), FORECAST_URL, state_short)
        for _, row in df.iterrows():
            temps[row["date"]] = (None if row["temp"] is None else round(float(row["temp"]), 2))
            source[row["date"]] = "forecast"

    # 3) beyond horizon (or any gaps) -> climatology
    remaining = [d for d in target_dates if temps.get(d) is None]
    if remaining:
        clim = _climatology_daily(coords, remaining, state_short, climatology_years, today)
        for d, t in clim.items():
            temps[d] = t
            source[d] = "climatology"

    # fill any leftover gaps with the mean of what we have (keep response clean)
    known = [v for v in temps.values() if v is not None]
    fallback = round(sum(known) / len(known), 2) if known else None
    for d in target_dates:
        if temps.get(d) is None:
            temps[d] = fallback
            source.setdefault(d, "fallback")

    return [
        {"date": d.isoformat(), "temperature_c": temps[d], "source": source.get(d, "fallback")}
        for d in target_dates
    ]
