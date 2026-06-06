from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pandas as pd
import requests
from power.utils.logger import get_logger

logger = get_logger("WeatherFetch")


# ---------------------------------------------------------------------------
# Weather response cache
#
# Every open-meteo call here is keyed by a *fixed* (state, date-range, endpoint)
# tuple, so the result for a given key is stable for the life of the process.
# Without this, the accuracy / forecast endpoints re-fetch the same days from
# open-meteo on every single request (and for CG that's a 4-district fan-out per
# day), which is what made `/api/power/forecast-accuracy` slow.
#
# The cache is a process-local, thread-safe TTL dict. Past (archive) weather is
# immutable so any TTL is safe; today/forecast can shift, so we keep a moderate
# TTL and refresh after it expires. Only successful (non-empty) results are
# cached, so a transient network failure is retried rather than stuck.
# ---------------------------------------------------------------------------
WEATHER_CACHE_TTL_SECONDS = 6 * 3600  # 6h: long enough to serve repeat calls, short enough to refresh the live forecast

_weather_cache: dict = {}
_weather_cache_lock = threading.Lock()


def _cache_get(key):
    with _weather_cache_lock:
        item = _weather_cache.get(key)
        if item is None:
            return None
        ts, value = item
        if time.time() - ts > WEATHER_CACHE_TTL_SECONDS:
            _weather_cache.pop(key, None)
            return None
        return value


def _cache_set(key, value):
    with _weather_cache_lock:
        _weather_cache[key] = (time.time(), value)


def _result_is_usable(value):
    """A result worth caching: a non-empty frame / a dict with any real value."""
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, dict):
        return any(v is not None for v in value.values())
    return value is not None


def _copy_result(value):
    """Hand back a defensive copy so callers can mutate freely (several do,
    e.g. the climatology proxy shifts the datetime column in place)."""
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict):
        return dict(value)
    return value


def _cached_call(key, producer):
    """Return the cached value for ``key`` or compute it via ``producer()``.

    Successful results are stored and every hand-off (store + return) is a copy,
    so the cached object is never mutated by a caller.
    """
    cached = _cache_get(key)
    if cached is not None:
        return _copy_result(cached)

    value = producer()
    if _result_is_usable(value):
        _cache_set(key, _copy_result(value))
    return value


def clear_weather_cache():
    """Drop every cached weather response (used by tests / manual refresh)."""
    with _weather_cache_lock:
        _weather_cache.clear()





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


#-------------------------------------
# Chhattisgarh district network.
#
# For CG, weather is NOT taken from the single STATE_COORDS["CG"] point. Instead
# we fetch the districts below concurrently and blend them into a single
# POPULATION-WEIGHTED average. This applies to every CG weather path that flows
# through the three low-level fetchers below (live/archive hourly, daily
# forecast, archive-proxy climatology, and the Climate-API normal), so live
# forecast and the Climate API both get the district-weighted value.
#
# The set is currently the 4 highest-load districts (weights sum to 1.0). The
# weighted average normalises by the weights of the districts that actually
# returned data, so it stays correct even if a district request fails, and the
# list can be grown/shrunk freely without touching the blending code below.
# Temperature is the primary target, but humidity/wind/rain are blended with the
# same weights to keep the CG record internally consistent.
# ---------------------------------------------------------------------------
CG_DISTRICTS = [
    {"name": "Raipur",   "lat": 21.2514, "lon": 81.6296, "weight": 0.40},  # Capital, max load
    {"name": "Bilaspur", "lat": 22.0796, "lon": 82.1391, "weight": 0.25},  # Industrial north
    {"name": "Korba",    "lat": 22.3595, "lon": 82.7501, "weight": 0.20},  # Power plant area
    {"name": "Jagdalpur","lat": 19.0748, "lon": 82.0388, "weight": 0.15},  # South CG, cooler temp
]
# Max concurrent district requests (thread pool over the blocking open-meteo
# `requests` calls — i.e. the districts are fetched in parallel, not serially).
# Kept modest so the fan-out doesn't trip open-meteo's burst rate limit (the
# Climate API is the strictest); the weighted average tolerates any district that
# still gets throttled by normalising over the weights that succeeded.
_DISTRICT_WORKERS = 8


def _parallel_districts(fetch_one):
    """Run ``fetch_one(district)`` for every CG district concurrently.

    Returns ``[(result, weight), ...]`` keeping only districts that returned a
    usable result (empty DataFrame / empty dict / None / errors are dropped).
    """
    out = []
    with ThreadPoolExecutor(max_workers=min(_DISTRICT_WORKERS, len(CG_DISTRICTS))) as ex:
        futures = {ex.submit(fetch_one, d): d for d in CG_DISTRICTS}
        for fut, dist in futures.items():
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.error(f"[CG/{dist['name']}] district fetch failed: {e}")
                continue
            if res is None:
                continue
            if isinstance(res, pd.DataFrame) and res.empty:
                continue
            if isinstance(res, dict) and not any(v is not None for v in res.values()):
                continue
            out.append((res, dist["weight"]))
    return out


def _weighted_average_df(frames_weights, key, value_cols):
    """Population-weighted average of ``value_cols`` across district frames.

    Frames are aligned on ``key`` (datetime or date); each row is normalised by
    the weights of the districts that have a (non-NaN) value at that key, so a
    missing district never biases the blend.
    """
    idx = sorted(set().union(*[set(df[key]) for df, _ in frames_weights]))
    num = {c: pd.Series(0.0, index=idx) for c in value_cols}
    den = {c: pd.Series(0.0, index=idx) for c in value_cols}

    for df, w in frames_weights:
        d = df.drop_duplicates(subset=[key]).set_index(key)
        for c in value_cols:
            s = pd.to_numeric(d[c], errors="coerce").reindex(idx)
            present = pd.Series(w, index=idx).where(s.notna(), 0.0)
            num[c] = num[c] + (w * s).fillna(0.0)
            den[c] = den[c] + present

    out = pd.DataFrame(index=pd.Index(idx, name=key))
    for c in value_cols:
        out[c] = num[c] / den[c].replace(0.0, pd.NA)
    return out.reset_index()


def _weighted_average_dict(dicts_weights, keys):
    """Population-weighted average of per-district ``{key: value}`` dicts.

    Normalised per key by the weights of districts with a non-None value.
    """
    out = {}
    for k in keys:
        num = den = 0.0
        for dct, w in dicts_weights:
            v = dct.get(k)
            if v is None:
                continue
            num += w * float(v)
            den += w
        out[k] = round(num / den, 2) if den > 0 else None
    return out


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
    """Range hourly weather -> 5-min df (cached per state/range/endpoint).

    For CG this fans out across every district (in parallel) and returns the
    population-weighted average; every other state is a single point request.
    """
    key = ("hourly", state_short, start_str, end_str, url, frequency)
    return _cached_call(
        key,
        lambda: _request_open_meteo_uncached(
            coords, start_str, end_str, url, state_short, frequency
        ),
    )


def _request_open_meteo_uncached(coords, start_str, end_str, url, state_short, frequency):
    if state_short == "CG":
        frames = _parallel_districts(
            lambda d: _request_open_meteo_single(
                d, start_str, end_str, url, f"CG/{d['name']}", frequency
            )
        )
        if not frames:
            return pd.DataFrame()
        out = _weighted_average_df(
            frames, "datetime",
            ["temperature_c", "humidity_pct", "wind_speed_ms", "rain_mm"],
        )
        out["state"] = "CG"
        out["frequency"] = frequency
        logger.info(
            f"[CG] population-weighted hourly weather from "
            f"{len(frames)}/{len(CG_DISTRICTS)} districts {start_str}..{end_str}"
        )
        return out

    return _request_open_meteo_single(coords, start_str, end_str, url, state_short, frequency)


def _request_open_meteo_single(coords, start_str, end_str, url, label, frequency):
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
            logger.info(f"[{label}] weather fetched {start_str}..{end_str}")
            break
        except Exception as e:
            if attempt == 2:
                logger.error(
                    f"[{label}] Weather fetch failed {start_str}..{end_str}: {e}"
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

    df["state"] = label
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
    """Daily mean/max temperature DataFrame (cached per state/range/endpoint).

    For CG this is the population-weighted blend of every district (fetched in
    parallel); every other state is a single point request.
    """
    key = ("daily", state_short, start_dt.isoformat(), end_dt.isoformat(), url)
    return _cached_call(
        key,
        lambda: _request_daily_uncached(coords, start_dt, end_dt, url, state_short),
    )


def _request_daily_uncached(coords, start_dt, end_dt, url, state_short):
    if state_short == "CG":
        frames = _parallel_districts(
            lambda d: _request_daily_single(d, start_dt, end_dt, url, f"CG/{d['name']}")
        )
        if not frames:
            return pd.DataFrame()
        out = _weighted_average_df(frames, "date", ["temp", "temp_max"])
        logger.info(
            f"[CG] population-weighted daily weather from "
            f"{len(frames)}/{len(CG_DISTRICTS)} districts {start_dt}..{end_dt}"
        )
        return out

    return _request_daily_single(coords, start_dt, end_dt, url, state_short)


def _request_daily_single(coords, start_dt, end_dt, url, label):
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
                    f"[{label}] Daily weather failed {start_dt}..{end_dt}: {e}"
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


# ---------------------------------------------------------------------------
# Open-Meteo Climate API (downscaled CMIP6, 1950-2050). Used to build a true
# multi-decade temperature normal for the days beyond the ~16-day forecast
# horizon (days ~17-30 of the 30-day forecast), which is far more stable than
# borrowing a single recent year from the archive.
#   https://climate-api.open-meteo.com/v1/climate
# ---------------------------------------------------------------------------
CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"
CLIMATE_NORMAL_YEARS = 30
CLIMATE_MODEL = "MRI_AGCM3_2_S"  # ~20 km high-res model, good coverage over India


def _climate_normal_daily(coords, target_dates, state_short, years=CLIMATE_NORMAL_YEARS):
    """
    `years`-year temperature normal for the given calendar days via the
    Open-Meteo Climate API (cached per state / day-set / horizon).

    For CG this is the population-weighted blend of every district (fetched in
    parallel); every other state is a single point request. Returns
    {date: temp_or_None}.
    """
    key = ("climate", state_short, tuple(sorted(target_dates)), years)
    return _cached_call(
        key,
        lambda: _climate_normal_daily_uncached(coords, target_dates, state_short, years),
    )


def _climate_normal_daily_uncached(coords, target_dates, state_short, years=CLIMATE_NORMAL_YEARS):
    if state_short == "CG":
        dicts = _parallel_districts(
            lambda d: _climate_normal_daily_single(d, target_dates, f"CG/{d['name']}", years)
        )
        if not dicts:
            return {d: None for d in target_dates}
        out = _weighted_average_dict(dicts, target_dates)
        filled = sum(1 for v in out.values() if v is not None)
        logger.info(
            f"[CG] population-weighted Climate-API {years}-yr normal from "
            f"{len(dicts)}/{len(CG_DISTRICTS)} districts filled "
            f"{filled}/{len(target_dates)} day(s)"
        )
        return out

    return _climate_normal_daily_single(coords, target_dates, state_short, years)


def _climate_normal_daily_single(coords, target_dates, label, years=CLIMATE_NORMAL_YEARS):
    """Single-point Climate-API normal. Returns {date: temp_or_None}."""
    end_year = datetime.now().year - 1
    start_year = end_year - (years - 1)
    md_start, md_end = min(target_dates), max(target_dates)

    def _safe(y, d):
        try:
            return date(y, d.month, d.day)
        except ValueError:  # e.g. Feb 29 in a non-leap proxy year
            return date(y, d.month, min(d.day, 28))

    range_start = _safe(start_year, md_start)
    range_end = _safe(end_year, md_end)

    params = {
        "latitude": round(coords["lat"], 4),
        "longitude": round(coords["lon"], 4),
        "start_date": range_start.isoformat(),
        "end_date": range_end.isoformat(),
        "models": CLIMATE_MODEL,
        "daily": "temperature_2m_mean",
        "timezone": "Asia/Kolkata",
    }

    data = None
    for attempt in range(3):
        try:
            r = requests.get(CLIMATE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 2:
                logger.error(
                    f"[{label}] Climate-API normal failed "
                    f"{range_start}..{range_end}: {e}"
                )
                return {d: None for d in target_dates}
            time.sleep(2)

    daily = (data or {}).get("daily", {})
    times = daily.get("time") or []
    # Single-model responses use the bare variable name; be tolerant of a
    # model-suffixed key (e.g. temperature_2m_mean_MRI_AGCM3_2_S) just in case.
    temp_key = next((k for k in daily if k.startswith("temperature_2m_mean")), None)
    temps_arr = daily.get(temp_key) if temp_key else None
    if not times or not temps_arr:
        logger.warning(f"[{label}] Climate-API returned no data {range_start}..{range_end}")
        return {d: None for d in target_dates}

    by_md = {}  # (month, day) -> [temps]
    for t, val in zip(times, temps_arr):
        if val is None:
            continue
        d = datetime.strptime(t, "%Y-%m-%d").date()
        by_md.setdefault((d.month, d.day), []).append(float(val))

    out = {}
    for d in target_dates:
        vals = by_md.get((d.month, d.day), [])
        out[d] = round(sum(vals) / len(vals), 2) if vals else None

    filled = sum(1 for v in out.values() if v is not None)
    logger.info(
        f"[{label}] Climate-API {years}-yr normal "
        f"({start_year}-{end_year}) filled {filled}/{len(target_dates)} day(s)"
    )
    return out


def fetch_daily_weather(state_short: str, from_date, days: int = 30, climatology_years: int = 3):
    """
    Daily mean temperature for `days` days starting at `from_date`.

    Returns a list aligned to the date range. Each item is a dict
    {"date", "temperature_c", "source"} where source is
    "forecast" | "archive" | "climatology_30yr" | "climatology" | "fallback".
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

    # 3) beyond horizon (days ~17-30, or any gaps) -> 30-year climate normal
    #    from the Open-Meteo Climate API. The legacy archive proxy is kept as a
    #    secondary fallback for any day the climate API can't fill.
    remaining = [d for d in target_dates if temps.get(d) is None]
    if remaining:
        clim = _climate_normal_daily(coords, remaining, state_short, CLIMATE_NORMAL_YEARS)
        for d, t in clim.items():
            if t is not None:
                temps[d] = t
                source[d] = "climatology_30yr"

        still_missing = [d for d in remaining if temps.get(d) is None]
        if still_missing:
            proxy = _climatology_daily(coords, still_missing, state_short, climatology_years, today)
            for d, t in proxy.items():
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
