"""
Shared configuration for the CG 5-min Temporal Fusion Transformer.

Keeping every name in one place guarantees that the training pipeline and the
inference pipeline build *exactly* the same feature set — the same idea behind
the XGBoost model's `feature_cols` attribute.
"""

# ----------------------------------------------------------------------
# Column roles
# ----------------------------------------------------------------------
TARGET = "y"
GROUP_COL = "series"          # single constant group ("CG") -> global model
TIME_IDX = "time_idx"

# Same inputs as the XGBoost model (weather + calendar + interactions + lags),
# expressed as TFT "known reals" (we know all of these for any future date).
#
#   weather      -> temperature_c, humidity_pct, rain_mm, wind_speed_ms
#   time         -> hour, is_peak, temp_x_hour, humidity_x_hour, wind_x_hour
#   calendar     -> is_weekend, is_holiday, season
#   daily shape  -> profile_y
#   seasonal lags-> y_lag_24h (288 steps), y_lag_7d (2016 steps)
#
# The short lags (y_lag_1..6 in XGBoost) are captured natively by the TFT
# encoder, which sees the full recent load trajectory — so they do not need to
# be passed as explicit columns.
KNOWN_REALS = [
    # weather
    "temperature_c", "humidity_pct", "rain_mm", "wind_speed_ms",
    # time
    "hour", "is_peak",
    "temp_x_hour", "humidity_x_hour", "wind_x_hour",
    # calendar
    "is_weekend", "is_holiday", "season",
    # daily behaviour + seasonal lags
    "profile_y", "y_lag_24h", "y_lag_7d",
]

# The load itself is the only "unknown" real — the encoder reads its history.
UNKNOWN_REALS = [TARGET]

# ----------------------------------------------------------------------
# Sequence geometry  (5-min resolution => 288 steps / day)
# ----------------------------------------------------------------------
STEPS_PER_DAY = 288
MAX_PREDICTION_LENGTH = STEPS_PER_DAY          # forecast one full day (288 slots)
MAX_ENCODER_LENGTH = STEPS_PER_DAY * 3         # 3 days of context for the encoder

FREQ = "5min"

# ----------------------------------------------------------------------
# Artifact naming
# ----------------------------------------------------------------------
MODEL_FILENAME = "state_5min_tft_CG.pkl"
SMOKE_FILENAME = "state_5min_tft_CG_smoke.pkl"
ARTIFACT_FORMAT = "tft-ckpt-v1"
