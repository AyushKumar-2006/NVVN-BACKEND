"""
TFT forecast router (additive — does not modify the existing power.api router).

Exposes:  GET /api/power/forecast-5min-tft?state_code=CHG&forecast_date=2026-07-03
"""

from ninja import Query, Router
from ninja.errors import HttpError

from power.api import MERIT_TO_SHORT_MAP
from power.schemas import DateQuerySchema, ForecastHourlyOut, StateShortEnum
from power.utils.forecast_tft import build_5min_tft_forecast_response

router = Router(tags=["TFT Forecast"])

# States that currently have a trained TFT model on disk.
TFT_SUPPORTED_STATES = {"CG"}


@router.get("/forecast-5min-tft", response=ForecastHourlyOut)
def forecast_5min_tft(
    request,
    state_code: StateShortEnum,
    query: DateQuerySchema = Query(...),
):
    """
    **URL:** GET /forecast-5min-tft
    **Description:** 5-minute load forecast from a Temporal Fusion Transformer
    (TFT) model. Uses the same inputs as the XGBoost model (weather, calendar,
    daily profile and seasonal lags); the TFT encoder additionally models the
    recent load trajectory directly.

    **Query Params:**
    - state_code: Short code of the state (e.g. CHG for Chhattisgarh)
    - forecast_date: YYYY-MM-DD (optional, defaults to today)

    **Note:** A trained TFT model currently exists only for CG (Chhattisgarh).
    Other states return HTTP 400 until their model is trained.

    **Response 200 OK Example:**
    ```json
    {
        "state": "Chhattisgarh",
        "date": "2026-07-03",
        "season": "monsoon",
        "weekday": "Friday",
        "is_weekend": false,
        "is_holiday": false,
        "energy_consumption_mu_per_day": 84.21,
        "average_load_mw": 3508.6,
        "peak_load_mw": 4123.9,
        "mape_difference_percent": null,
        "points": [
            {"datetime": "2026-07-03T00:00:00", "mw": 3290.4, "temperature": 27.1},
            {"datetime": "2026-07-03T00:05:00", "mw": 3285.7, "temperature": 27.1}
        ]
    }
    ```
    """
    short = MERIT_TO_SHORT_MAP.get(state_code.value, state_code.value)

    if short not in TFT_SUPPORTED_STATES:
        raise HttpError(
            400,
            f"No TFT model trained for '{short}'. "
            f"Available: {sorted(TFT_SUPPORTED_STATES)}.",
        )

    return build_5min_tft_forecast_response(state=short, forecast_date=query.forecast_date)
