import pandas as pd

# import os
# import pandas as pd
# from power.ml.models.xgb_model import train_xgb_model
# from power.ml.progress import log_progress
# from power.ml.trainy.common import clean_outliers, merge_live_weather
# from power.models import StateLoad5Min
# from power.utils.logger import get_logger
# from power.utils.metadata import add_calendar_features

# TMP_DIR = "./power/ml/tmp"
# os.makedirs(TMP_DIR, exist_ok=True)




# ----------------------------------
# PEAK + INTERACTION FEATURES
# ----------------------------------
def add_peak_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    df["is_peak"] = df["hour"].between(18, 23).astype(int)

    df["temp_x_hour"] = df["temperature_c"] * df["hour"]
    df["humidity_x_hour"] = df["humidity_pct"] * df["hour"]
    df["wind_x_hour"] = df["wind_speed_ms"] * df["hour"]
    return df






# ----------------------------------
# SAMPLE WEIGHT
# ----------------------------------
def add_sample_weight(df: pd.DataFrame) -> pd.DataFrame:
    df["sample_weight"] = df["is_peak"].map(lambda x: 3.0 if x == 1 else 1.0)
    return df






# # ----------------------------------
# # TRAIN FUNCTION
# # ----------------------------------
# def train_state_5min_model(state: str):
#     logger = get_logger(f"TRAIN-{state}")
#     weather_tmp = f"{TMP_DIR}/{state}_5min_weather.parquet"

#     log_progress(logger, "Fetching raw data", "xgboost", 5)

#     raw = pd.DataFrame(
#         StateLoad5Min.objects
#         .filter(state=state)
#         .values("datetime", "load_mw", "brpl", "bypl", "ndpl", "ndmc", "mes")
#         .order_by("datetime")
#     )

#     if raw.empty:
#         raise ValueError(f"No data for state={state}")
    
#     raw = raw.sort_values("datetime")
#     raw["datetime"] = pd.to_datetime(raw["datetime"])

#     # -----------------------------
#     # TARGET
#     # -----------------------------
#     discoms = ["brpl", "bypl", "ndpl", "ndmc", "mes"]
#     raw["y"] = raw["load_mw"]
#     raw.loc[raw["y"].isna(), "y"] = raw[discoms].sum(axis=1)

#     # -----------------------------
#     # RESAMPLE
#     # -----------------------------
#     df = (
#         raw.set_index("datetime")
#         .resample("5min")
#         .mean()
#         .interpolate()
#         .reset_index()
#         .rename(columns={"datetime": "ds"})
#     )

#     # -----------------------------
#     # WEATHER
#     # -----------------------------
#     log_progress(logger, "Preparing weather", "xgboost", 30)

#     if os.path.exists(weather_tmp):
#         weather = pd.read_parquet(weather_tmp)
#     else:
#         weather = merge_live_weather(
#             start_date=df["ds"].min().date(),
#             end_date=df["ds"].max().date(),
#             state=state,
#             frequency="hourly",
#         )
#         weather.to_parquet(weather_tmp)

#     df = pd.merge_asof(
#         df.sort_values("ds"),
#         weather.sort_values("ds"),
#         on="ds",
#         direction="nearest",
#         tolerance=pd.Timedelta("1h"),
#     )

#     # -----------------------------
#     # CALENDAR + PEAK
#     # -----------------------------
#     df = add_calendar_features(df)
#     df = add_peak_features(df)

#     # -----------------------------
#     # SLOT + DATE PROFILE (🔥 CORE FIX)
#     # -----------------------------
#     df["slot"] = df["ds"].dt.hour * 12 + df["ds"].dt.minute // 5
#     df["mmdd"] = df["ds"].dt.strftime("%m-%d")

#     profile = (
#         df.groupby(["mmdd", "slot"])["y"]
#         .mean()
#         .reset_index()
#         .rename(columns={"y": "profile_y"})
#     )

#     df = df.merge(profile, on=["mmdd", "slot"], how="left")

#     # -----------------------------
#     # OUTLIERS
#     # -----------------------------
#     df = clean_outliers(df)

#     # -----------------------------
#     # SHORT LAGS ONLY
#     # -----------------------------
#     for i in range(1, 7):  # last 30 minutes
#         df[f"y_lag_{i}"] = df["y"].shift(i)

#     df["y_lag_24h"] = df["y"].shift(288)
#     df["y_lag_7d"] = df["y"].shift(2016)

#     df = add_sample_weight(df)
#     df = df.dropna().reset_index(drop=True)

#     print("TRAIN ROWS:", len(df), "COLS:", df.shape[1])

#     # -----------------------------
#     # TRAIN
#     # -----------------------------
#     log_progress(logger, "Training started", "xgboost", 60)
#     model = train_xgb_model(df,)
#     log_progress(logger, "Training completed", "xgboost", 100)

#     return model

#ye actual h
import os
import pandas as pd
from power.ml.models.xgb_model import train_xgb_model
from power.ml.progress import log_progress
from power.ml.trainy.common import clean_outliers, merge_live_weather
from power.models import StateLoad5Min, Weather
from power.utils.logger import get_logger
from power.utils.metadata import add_calendar_features


# ----------------------------------
# TRAIN FUNCTION
# ----------------------------------
def train_state_5min_model(state: str):
    logger = get_logger(f"TRAIN-{state}")

    log_progress(logger, "Fetching raw data", "xgboost", 5)

    raw = pd.DataFrame(
        StateLoad5Min.objects
        .filter(state=state)
        .values("datetime", "load_mw", "brpl", "bypl", "ndpl", "ndmc", "mes")
        .order_by("datetime")
    )

    if raw.empty:
        raise ValueError(f"No data for state={state}")
    
    raw = raw.sort_values("datetime")
    raw["datetime"] = pd.to_datetime(raw["datetime"])

    # -----------------------------
    # TARGET
    # -----------------------------
    discoms = ["brpl", "bypl", "ndpl", "ndmc", "mes"]
    raw["y"] = raw["load_mw"]
    raw.loc[raw["y"].isna(), "y"] = raw[discoms].sum(axis=1)

    # -----------------------------
    # RESAMPLE
    # -----------------------------
    # Only the target `y` is needed downstream; selecting it keeps the
    # resample numeric (discom columns are all-NULL/object dtype for states
    # like CG and would otherwise break .interpolate()).
    df = (
        raw.set_index("datetime")[["y"]]
        .resample("5min")
        .mean()
        .interpolate()
        .reset_index()
        .rename(columns={"datetime": "ds"})
    )

    # -----------------------------
    # WEATHER (DB BASED)
    # -----------------------------
    log_progress(logger, "Preparing weather", "xgboost", 30)

    start_dt = df["ds"].min()
    end_dt = df["ds"].max()

    weather_qs = Weather.objects.filter(
        state=state,
        datetime__gte=start_dt,
        datetime__lte=end_dt,
    ).order_by("datetime")

    # Agar DB me weather nahi hai to fetch karo aur save karo
    if not weather_qs.exists():

        weather = merge_live_weather(
            start_date=start_dt.date(),
            end_date=end_dt.date(),
            state=state,
            frequency="hourly",
        )

        weather_objects = [
            Weather(
                state=state,
                datetime=row["ds"],
                frequency="hourly",
                temperature_c=row["temperature_c"],
                humidity_pct=row.get("humidity_pct"),
                rain_mm=row.get("rain_mm"),
                wind_speed_ms=row.get("wind_speed_ms"),
                source="open-meteo",
            )
            for _, row in weather.iterrows()
        ]

        Weather.objects.bulk_create(weather_objects, ignore_conflicts=True)

    # DB se weather load karo
    weather = pd.DataFrame(
        Weather.objects.filter(
            state=state,
            datetime__gte=start_dt,
            datetime__lte=end_dt,
        )
        .order_by("datetime")
        .values(
            "datetime",
            "temperature_c",
            "humidity_pct",
            "rain_mm",
            "wind_speed_ms",
        )
    )



    weather = weather.rename(columns={"datetime": "ds"})
    weather["ds"] = pd.to_datetime(weather["ds"])

    df = pd.merge_asof(
        df.sort_values("ds"),
        weather.sort_values("ds"),
        on="ds",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    # -----------------------------
    # CALENDAR + PEAK
    # -----------------------------
    df = add_calendar_features(df)
    df = add_peak_features(df)

    # -----------------------------
    # SLOT + DATE PROFILE
    # -----------------------------
    df["slot"] = df["ds"].dt.hour * 12 + df["ds"].dt.minute // 5
    df["mmdd"] = df["ds"].dt.strftime("%m-%d")

    profile = (
        df.groupby(["mmdd", "slot"])["y"]
        .mean()
        .reset_index()
        .rename(columns={"y": "profile_y"})
    )

    df = df.merge(profile, on=["mmdd", "slot"], how="left")

    # -----------------------------
    # OUTLIERS
    # -----------------------------
    df = clean_outliers(df)

    # -----------------------------
    # SHORT LAGS ONLY
    # -----------------------------
    for i in range(1, 7):
        df[f"y_lag_{i}"] = df["y"].shift(i)

    df["y_lag_24h"] = df["y"].shift(288)
    df["y_lag_7d"] = df["y"].shift(2016)

    df = add_sample_weight(df)
    df = df.dropna().reset_index(drop=True)

    print("TRAIN ROWS:", len(df), "COLS:", df.shape[1])

    # -----------------------------
    # TRAIN
    # -----------------------------
    log_progress(logger, "Training started", "xgboost", 60)
    model = train_xgb_model(df)
    log_progress(logger, "Training completed", "xgboost", 100)

    return model




