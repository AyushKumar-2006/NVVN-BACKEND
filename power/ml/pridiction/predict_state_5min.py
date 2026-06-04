# import pandas as pd
# from datetime import timedelta
# from power.utils.metadata import add_calendar_features
# from power.ml.trainy.common import merge_live_weather
# from power.ml.model_store import load_model
# from power.models import StateLoad5Min


# def predict_state_5min_data(state: str, forecast_date: str):
#     print("\n================= START PREDICTION =================\n")

#     model = load_model(f"state_5min_{state}.pkl")

#     # 🔥 ONLY FIX 1: force local midnight (IST-naive)
#     forecast_date = pd.to_datetime(forecast_date).normalize()

#     # --------------------------------------------------
#     # 1️⃣ Time grid (LOCAL IST)
#     # --------------------------------------------------
#     df = pd.DataFrame({
#         "ds": pd.date_range(
#             forecast_date,
#             forecast_date + timedelta(days=1) - timedelta(minutes=5),
#             freq="5min"
#         )
#     })

#     df["slot"] = df["ds"].dt.hour * 12 + df["ds"].dt.minute // 5
#     df["mmdd"] = df["ds"].dt.strftime("%m-%d")

#     # --------------------------------------------------
#     # 2️⃣ Historical profile (SAME LOGIC, LOCAL TIME)
#     # --------------------------------------------------
#     hist = pd.DataFrame(
#         StateLoad5Min.objects
#         .filter(
#             state=state,
#             datetime__month=forecast_date.month,
#             datetime__day=forecast_date.day
#         )
#         .values("datetime", "load_mw", "brpl", "bypl", "ndpl", "ndmc", "mes")
#     )

#     # 🔥 ONLY FIX 2: treat DB datetime as local IST (naive)
#     hist["datetime"] = pd.to_datetime(hist["datetime"])

#     hist["y"] = hist["load_mw"].fillna(
#         hist[["brpl", "bypl", "ndpl", "ndmc", "mes"]].sum(axis=1)
#     )

#     hist["date"] = hist["datetime"].dt.date
#     hist["slot"] = hist["datetime"].dt.hour * 12 + hist["datetime"].dt.minute // 5

#     # 🔍 PRINT 1: WHICH DATES ARE USED
#     print("\n========== HISTORICAL DATES USED ==========")
#     print(sorted(hist["date"].unique()))

#     # 🔍 PRINT 2: FULL 24H RAW DATA (first & last)
#     print("\n========== RAW HISTORICAL 24H DATA (SAMPLE) ==========")
#     print(hist[["datetime", "y"]].head(20))
#     print("...")
#     print(hist[["datetime", "y"]].tail(20))
#     print("TOTAL HIST ROWS:", len(hist))

#     # --------------------------------------------------
#     # Slot-wise average (profile)
#     # --------------------------------------------------
#     profile = (
#         hist.groupby("slot")["y"]
#         .mean()
#         .reset_index()
#         .rename(columns={"y": "profile_y"})
#     )

#     # 🔍 PRINT 3: SLOT-WISE 24H PROFILE
#     print("\n========== SLOT-WISE 24H AVERAGE PROFILE ==========")
#     print(profile.head(20))
#     print("...")
#     print(profile.tail(20))
#     print("TOTAL SLOTS:", len(profile))  # should be 288

#     df = df.merge(profile, on="slot", how="left")

#     print("PROFILE SAMPLE:")
#     print(df[["ds", "slot", "profile_y"]].head(10))

#     # --------------------------------------------------
#     # 3️⃣ Weather (LOCAL DATE)
#     # --------------------------------------------------
#     weather = merge_live_weather(
#         start_date=forecast_date.date(),
#         end_date=forecast_date.date(),
#         state=state,
#         frequency="hourly",
#     )

#     df = pd.merge_asof(
#         df.sort_values("ds"),
#         weather.sort_values("ds"),
#         on="ds",
#         direction="nearest",
#         tolerance=pd.Timedelta("1h"),
#     )

#     # --------------------------------------------------
#     # 4️⃣ Calendar + peak
#     # --------------------------------------------------
#     df = add_calendar_features(df)

#     df["hour"] = df["ds"].dt.hour
#     df["is_peak"] = df["hour"].between(18, 23).astype(int)

#     df["temp_x_hour"] = df["temperature_c"] * df["hour"]
#     df["humidity_x_hour"] = df["humidity_pct"] * df["hour"]
#     df["wind_x_hour"] = df["wind_speed_ms"] * df["hour"]

#     # --------------------------------------------------
#     # 5️⃣ Lag construction (SAME LOGIC)
#     # --------------------------------------------------
#     for i in range(1, 7):
#         df[f"y_lag_{i}"] = df["profile_y"]

#     df["y_lag_24h"] = df["profile_y"]
#     df["y_lag_7d"] = df["profile_y"]

#     # --------------------------------------------------
#     # 6️⃣ Feature validation
#     # --------------------------------------------------
#     missing = set(model.feature_cols) - set(df.columns)
#     if missing:
#         raise ValueError(f"Missing features: {missing}")

#     # --------------------------------------------------
#     # 7️⃣ Predict
#     # --------------------------------------------------
#     X = df[model.feature_cols]
#     df["mw"] = model.predict(X)

#     print("\n================= FINAL OUTPUT =================\n")
#     print(df[["ds", "mw", "temperature_c"]].head(20))
#     print(df[["ds", "mw", "temperature_c"]].tail(20))

#     return df[["ds", "mw", "temperature_c"]]





#peak load manage dynamically
import pandas as pd
from datetime import timedelta
from power.utils.metadata import add_calendar_features
from power.ml.trainy.common import merge_live_weather
from power.ml.model_store import load_model
from power.models import StateLoad5Min


def predict_state_5min_data(state: str, forecast_date: str):

    print("\n================= START PREDICTION =================\n")

    model = load_model(f"state_5min_{state}.pkl")

    # force local midnight
    forecast_date = pd.to_datetime(forecast_date).normalize()

    # --------------------------------------------------
    # TIME GRID
    # --------------------------------------------------
    df = pd.DataFrame({
        "ds": pd.date_range(
            forecast_date,
            forecast_date + timedelta(days=1) - timedelta(minutes=5),
            freq="5min"
        )
    })

    df["slot"] = df["ds"].dt.hour * 12 + df["ds"].dt.minute // 5
    df["mmdd"] = df["ds"].dt.strftime("%m-%d")


    # --------------------------------------------------
    # HISTORICAL DATA (same logic as your original)
    # --------------------------------------------------
    hist = pd.DataFrame(
        StateLoad5Min.objects
        .filter(
            state=state,
            datetime__month=forecast_date.month,
            datetime__day=forecast_date.day
        )
        .values(
            "datetime",
            "load_mw",
            "brpl",
            "bypl",
            "ndpl",
            "ndmc",
            "mes"
        )
    )

    if hist.empty:
        raise ValueError("No historical data found")

    hist["datetime"] = pd.to_datetime(hist["datetime"])


    # --------------------------------------------------
    # BUILD TARGET
    # --------------------------------------------------
    hist["y"] = hist["load_mw"].fillna(
        hist[["brpl", "bypl", "ndpl", "ndmc", "mes"]].sum(axis=1)
    )


    # --------------------------------------------------
    # DYNAMIC OUTLIER REMOVAL (NO HARDCODE)
    # Uses statistical IQR method
    # --------------------------------------------------
    Q1 = hist["y"].quantile(0.25)
    Q3 = hist["y"].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    hist = hist[(hist["y"] >= lower) & (hist["y"] <= upper)]


    # --------------------------------------------------
    # SLOT CREATION
    # --------------------------------------------------
    hist["slot"] = hist["datetime"].dt.hour * 12 + hist["datetime"].dt.minute // 5


    print("\n========== HISTORICAL DATES USED ==========")
    print(sorted(hist["datetime"].dt.date.unique()))


    # --------------------------------------------------
    # PROFILE (median instead of mean)
    # This fixes spike dynamically
    # --------------------------------------------------
    profile = (
        hist.groupby("slot")["y"]
        .median()
        .reset_index()
        .rename(columns={"y": "profile_y"})
    )


    print("\nPROFILE MAX:", profile["profile_y"].max())
    print("PROFILE MIN:", profile["profile_y"].min())


    df = df.merge(profile, on="slot", how="left")


    # safety fill (dynamic)
    df["profile_y"] = df["profile_y"].interpolate().ffill().bfill()


    # --------------------------------------------------
    # WEATHER
    # --------------------------------------------------
    weather = merge_live_weather(
        start_date=forecast_date.date(),
        end_date=forecast_date.date(),
        state=state,
        frequency="hourly",
    )


    df = pd.merge_asof(
        df.sort_values("ds"),
        weather.sort_values("ds"),
        on="ds",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )


    # --------------------------------------------------
    # CALENDAR FEATURES
    # --------------------------------------------------
    df = add_calendar_features(df)

    df["hour"] = df["ds"].dt.hour

    df["is_peak"] = df["hour"].between(18, 23).astype(int)

    df["temp_x_hour"] = df["temperature_c"] * df["hour"]

    df["humidity_x_hour"] = df["humidity_pct"] * df["hour"]

    df["wind_x_hour"] = df["wind_speed_ms"] * df["hour"]


    # --------------------------------------------------
    # LAGS (same logic)
    # --------------------------------------------------
    for i in range(1, 7):

        df[f"y_lag_{i}"] = df["profile_y"]

    df["y_lag_24h"] = df["profile_y"]

    df["y_lag_7d"] = df["profile_y"]


    # --------------------------------------------------
    # FEATURE VALIDATION
    # --------------------------------------------------
    missing = set(model.feature_cols) - set(df.columns)

    if missing:
        raise ValueError(f"Missing features: {missing}")


    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------
    X = df[model.feature_cols]

    df["mw"] = model.predict(X)


    print("\n================= FINAL OUTPUT =================\n")

    print("Prediction max:", df["mw"].max())

    print("Prediction min:", df["mw"].min())


    return df[["ds", "mw", "temperature_c"]]












