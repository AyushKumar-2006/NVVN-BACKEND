"""
Additive comparison endpoints (predicted vs actual / accuracy).

Kept in a separate router so the existing power/api.py is left untouched. The
router is mounted under /api/power/ from config/urls.py.
"""

from datetime import date as date_type

from ninja import Query, Router

from power.api import MERIT_TO_SHORT_MAP
from power.schemas import StateShortEnum
from power.utils.compare import build_forecast_accuracy, build_forecast_compare
from power.utils.dashboard import build_daily_dashboard

router = Router()


@router.get("/forecast-compare", response={200: dict})
def forecast_compare(request, state_code: StateShortEnum, date: date_type = Query(...)):
    """
    **URL:** GET /forecast-compare
    **Description:** Compares the model's predicted load for a date against the
    actual recorded load (when available).

    **Query Params:**
    - state_code: Short code of the state (Dropdown)
    - date: YYYY-MM-DD

    **Returns:** predicted vs actual (average / peak / energy), the difference in
    MW and %, plus a slot-level MAPE and accuracy score for that day. If no
    actual data exists for the date, the prediction is returned on its own.
    """
    short = MERIT_TO_SHORT_MAP.get(state_code.value, state_code.value)
    return build_forecast_compare(state=short, day=date)


@router.get("/forecast-accuracy", response={200: dict})
def forecast_accuracy(request, state_code: StateShortEnum, days: int = Query(30)):
    """
    **URL:** GET /forecast-accuracy
    **Description:** Day-by-day predicted vs actual accuracy over the most recent
    `days` days that have actual data, plus an overall MAPE / accuracy score.

    **Query Params:**
    - state_code: Short code of the state (Dropdown)
    - days: number of recent days to evaluate (default 30, max 120)
    """
    short = MERIT_TO_SHORT_MAP.get(state_code.value, state_code.value)
    return build_forecast_accuracy(state=short, days=days)


@router.get("/daily-dashboard", response={200: dict})
def daily_dashboard(request, state_code: StateShortEnum):
    """
    **URL:** GET /daily-dashboard
    **Description:** Unified daily view combining recent actuals and the upcoming
    forecast for a state.

    **Query Params:**
    - state_code: Short code of the state (Dropdown)

    **Returns** a `days` timeline:
    - PAST days (status "actual"): actual vs XGBoost-predicted load + accuracy_pct
    - FUTURE days (status "forecast"): Prophet predicted load / peak / energy,
      risk_level (NORMAL / HIGH RISK / CRITICAL), and a power-requirement insight
      (how_much_needed_mw, shortage_risk, recommended_reserve_mw = peak x 1.15)
    """
    short = MERIT_TO_SHORT_MAP.get(state_code.value, state_code.value)
    return build_daily_dashboard(state=short)
