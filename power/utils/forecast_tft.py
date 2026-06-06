"""
Response builder for the TFT 5-min forecast endpoint.

Produces the exact same payload shape as the XGBoost `build_5min_forecast_response`
(ForecastHourlyOut schema) so the frontend can consume either interchangeably.

Note: unlike the XGBoost builder this does NOT write to DailyPredictionHistory,
to avoid overwriting the existing model's saved predictions.
"""

from datetime import date

from ninja.errors import HttpError

from power.utils.metadata import STATE_CODE_TO_NAME, day_metadata
from power.ml.pridiction.predict_state_5min_tft import predict_state_5min_tft


def build_5min_tft_forecast_response(state: str, forecast_date):
    try:
        if isinstance(forecast_date, date):
            forecast_date_obj = forecast_date
        elif isinstance(forecast_date, str):
            forecast_date_obj = date.fromisoformat(forecast_date.strip())
        else:
            raise ValueError("forecast_date must be str or date")

        meta = day_metadata(forecast_date_obj)

        forecast_df = predict_state_5min_tft(state=state, forecast_date=forecast_date_obj)

        if forecast_df.empty:
            raise ValueError("Empty TFT forecast data")

        forecast_df["temperature_c"] = (
            forecast_df["temperature_c"]
            .astype(float)
            .ffill()
            .bfill()
            .fillna(25)
            .round(2)
        )
        forecast_df["mw"] = forecast_df["mw"].astype(float).round(2)

        points = [
            {
                "datetime": row.ds.isoformat(),
                "mw": row.mw,
                "temperature": row.temperature_c,
            }
            for _, row in forecast_df.iterrows()
        ]

        loads = forecast_df["mw"]
        energy_mwh = loads.sum() * (5 / 60)

        return {
            "state": STATE_CODE_TO_NAME.get(state, state),
            "date": forecast_date_obj.isoformat(),
            **meta,
            "energy_consumption_mu_per_day": round(energy_mwh / 1000, 2),
            "average_load_mw": round(loads.mean(), 2),
            "peak_load_mw": round(loads.max(), 2),
            "mape_difference_percent": None,
            "points": points,
        }

    except HttpError:
        raise
    except Exception as e:
        raise HttpError(400, str(e))
