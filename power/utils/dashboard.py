"""
Combined daily dashboard (additive feature).

Powers:  GET /api/power/daily-dashboard?state_code=CHG

Stitches together the two existing capabilities — no new modelling logic:
  - PAST days  : XGBoost 5-min predicted vs actual + accuracy
                 (reuses power.utils.compare.build_forecast_accuracy)
  - FUTURE days: Prophet 30-day daily forecast + risk level
                 (reuses power.ml.pridiction.predict_state_30day)
and layers a simple power-requirement insight on top of the forecast days.
"""

from datetime import date

from ninja.errors import HttpError

from power.ml.pridiction.predict_state_30day import predict_state_30day_data
from power.utils.compare import build_forecast_accuracy
from power.utils.metadata import STATE_CODE_TO_NAME

PAST_DAYS = 30
RESERVE_BUFFER = 1.15  # 15% safety reserve


def build_daily_dashboard(state: str) -> dict:
    try:
        # ---- PAST: most recent PAST_DAYS days that have actual data ----
        accuracy = build_forecast_accuracy(state, PAST_DAYS)

        # ---- FUTURE: next 30 days from today (Prophet) ----
        forecast = predict_state_30day_data(state, date.today())
        historical_peak = float(forecast["historical_peak_mw"])

        days = []

        for r in accuracy["daily"]:
            if r.get("accuracy_pct") is None:  # skip any day that failed to evaluate
                continue
            days.append({
                "date": r["date"],
                "status": "actual",
                "actual_mw": r["actual_avg_mw"],
                "predicted_mw": r["predicted_avg_mw"],
                "actual_peak_mw": r["actual_peak_mw"],
                "predicted_peak_mw": r["predicted_peak_mw"],
                "accuracy_pct": r["accuracy_pct"],
                "error_pct": r["error_pct"],
            })

        for fd in forecast["days"]:
            peak = float(fd["predicted_peak_load_mw"])
            avg = float(fd["predicted_average_load_mw"])
            days.append({
                "date": fd["date"],
                "status": "forecast",
                "predicted_load_mw": round(avg, 2),
                "predicted_peak_mw": round(peak, 2),
                "predicted_energy_mu": round(avg * 24 / 1000, 2),
                "risk_level": fd["risk_level"],
                "temperature_c": fd["temperature_c"],
                # ---- power requirement insight ----
                "how_much_needed_mw": round(peak, 2),
                "shortage_risk": bool(peak > historical_peak),
                "recommended_reserve_mw": round(peak * RESERVE_BUFFER, 2),
            })

        past_count = sum(1 for d in days if d["status"] == "actual")
        forecast_count = sum(1 for d in days if d["status"] == "forecast")

        return {
            "state": STATE_CODE_TO_NAME.get(state, state),
            "state_code": state,
            "historical_peak_mw": round(historical_peak, 2),
            "past_days": past_count,
            "forecast_days": forecast_count,
            "total_days": len(days),
            "days": days,
        }

    except HttpError:
        raise
    except Exception as e:
        raise HttpError(400, str(e))
