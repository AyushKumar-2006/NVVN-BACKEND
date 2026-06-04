from threading import Thread
from ninja import Query, Router, File
from ninja.files import UploadedFile
import pandas as pd
from power.ml.disaggregate import disaggregate_hourly_to_15min
from power.ml.manage_models import train_all_models
# from power.ml.weather import fetch_and_save_weather
from power.models import DailyPredictionHistory, Weather
from power.schemas import DateQuerySchema, Forecast15MinOut, ForecastHourlyOut, PreviousPredictionItem, PreviousPredictionOut, StateOut, StateShortEnum, TemperatureOut
from power.utils.forecast import build_5min_forecast_response, generate_all_forecasts
from power.utils.helper import build_load_forecast_response
from power.utils.metadata import STATE_CODE_TO_NAME, STATE_IN
from power.utils.upload import save_power_data_from_xlsx
from ninja.errors import HttpError
from django.db.models import Avg
from power.utils.validation import validate_date
# from power.utils.forecast import  get_forecast_5min_data, get_hourly_forecast_data
from typing import Optional, List
from ninja.pagination import paginate, PageNumberPagination
from ninja.errors import HttpError
import requests
from power.schemas import MeritStateCurrentOut
from datetime import date
from power.ml.pridiction.predict_state_30day import build_30day_forecast_response




MERIT_TO_SHORT_MAP = {
    "AP":  "AP",
    "ACP": "AP",
    "ASM": "AS",
    "BHR": "BR",
    "CHG": "CG",
    "CTG": "CH",
    "DNH": "DNH",
    "DND": "DND",
    "DL":  "DL",
    "GOA": "GA",
    "GJT": "GJ",
    "HRN": "HR",
    "HP":  "HP",
    "JAK": "JK",
    "JHK": "JH",
    "KRT": "KA",
    "KRL": "KL",
    "MPD": "MP",
    "MHA": "MH",
    "MIP": "MN",
    "MGA": "ML",
    "MZM": "MZ",
    "NGD": "NL",
    "ODI": "OD",
    "PU":  "PY",
    "PNB": "PB",
    "RJ":  "RJ",
    "SKM": "SK",
    "TND": "TN",
    "TLG": "TS",
    "TPA": "TR",
    "UP":  "UP",
    "UTK": "UK",
    "BGL": "WB",
}




router = Router()





@router.post("/upload-xlsx")
def upload_xlsx(request, file: UploadedFile = File(...)):
    """
    **URL:** POST /upload-xlsx  
    **Description:** Upload XLSX file containing historical electricity data. Automatically saves to DB and retrains ML models.  

    **Payload:**
    - file: XLSX file

    **Response 200 OK:**
    ```json
    {
        "status": "success",
        "rows_inserted": 123,
        "ml_status": "retrained"
    }
    ```

    **Error 400:** Invalid file or missing columns
    """

    try:
        rows_inserted = save_power_data_from_xlsx(file)
    except ValueError as e:
        raise HttpError(status_code=400, message=f"{e}")

    # # Auto-train ML model
    # train_model()

    return {
        "status": "success",
        "rows_inserted": rows_inserted,
        "ml_status": "retrained"
    }











@router.post("/train-all-models/", response={200: dict})
def train_all_models_api(request):
    """
    **URL:** POST /train-all-models/  
    **Description:** Trains all ML models for all states and regions in background.  

    **Response 200 OK:**
    ```json
    {
        "message": "Model training has started in the background. Check logs for progress."
    }
    ```
    """
    train_all_models() 
    return {"message": "Model training has started in the background. Check logs for progress."}







@router.get("/states/in", response=List[StateOut])
def list_states_in(request):
    """
    **URL:** GET /states/in  
    **Description:** Returns list of all Indian states with short code and full name.  

    **Response 200 OK:**
    ```json
    [
        {"code": "DL", "name": "Delhi"},
        {"code": "MH", "name": "Maharashtra"}
    ]
    ```
    """

    return STATE_IN





# @router.get("/forecast-hourly", response=ForecastHourlyOut)
# def forecast_hourly(request, state_code: StateShortEnum, query: DateQuerySchema = Query(...)):
#     """
#     **URL:** GET /forecast-hourly  
#     **Description:** Returns hourly forecast for a given state and date.  

#     **Query Params:**
#     - state_code: Short code of the state (Dropdown)
#     -- example: DL, MH, TN, UP, AP, AR, AS, BR, CH, CG, GA, GJ, HR, HP, JK, JH, KA, KL, MN, ML, MZ, MP, NL, OD, PY, PB, RJ, SK, TS, TR, UK, WB
#     - forecast_date: YYYY-MM-DD (optional, defaults to today)

#     **Response 200 OK Example:**
#     ```json
#     {
#         "state": "West Bengal",
#         "date": "2026-01-09",
#         "season": "Winter",
#         "weekday": "Monday",
#         "is_weekend": false,
#         "is_holiday": false,
#         "energy_consumption_mu_per_day": 1234.56,
#         "average_load_mw": 500.25,
#         "peak_load_mw": 750.40,
#         "mape_difference_percent": 3.5,
#         "points": [
#             {"datetime": "2026-01-09T00:00:00", "mw": 480.5, "temperature": 18.2}
#         ]
#     }
#     ```
#     """
#     data = get_hourly_forecast_data(state_code, query.forecast_date)
#     data["state"] = STATE_CODE_TO_NAME.get(state_code.value, state_code.value)
#     return data






# @router.get("/forecast-15min", response=Forecast15MinOut)
# def forecast_15min(
#     request,
#     state_code: StateShortEnum,
#     query: DateQuerySchema = Query(...)
# ):
#     """
#     **URL:** GET /forecast-15min  
#     **Description:** Returns 15-minute forecast by disaggregating hourly forecast.  

#     - state_code: Short code of the state (Dropdown)
#     -- example: DL, MH, TN, UP, AP, AR, AS, BR, CH, CG, GA, GJ, HR, HP, JK, JH, KA, KL, MN, ML, MZ, MP, NL, OD, PY, PB, RJ, SK, TS, TR, UK, WB
#     - forecast_date: YYYY-MM-DD (optional, defaults to today)

#     **Response 200 OK Example:**
#     ```json
#     {
#         "state": "WB",
#         "date": "2026-01-09",
#         "points": [
#             {"datetime": "2026-01-09T00:00:00", "mw": 480.5},
#             {"datetime": "2026-01-09T00:15:00", "mw": 485.2}
#         ]
#     }
#     ```
#     """

#     forecast_date = query.forecast_date

#     hourly = get_hourly_forecast_data(state_code, forecast_date)

#     df = pd.DataFrame(hourly["points"])
#     df["ds"] = pd.to_datetime(df["datetime"])
#     df["yhat"] = df["mw"]

#     df_15 = disaggregate_hourly_to_15min(df)

#     points = [
#         {
#             "datetime": row.ds.isoformat(),
#             "mw": round(row.yhat, 2)
#         }
#         for row in df_15.itertuples()
#     ]

#     return {
#         "state": hourly["state"],
#         "date": forecast_date.isoformat(),
#         "points": points
#     }








@router.get("/forecast-5min", response=ForecastHourlyOut)
def forecast_5min(request, state_code: StateShortEnum, query: DateQuerySchema = Query(...)):
    """
    **URL:** GET /forecast-5min  
    **Description:** Returns 5-minute forecast for a given state and date.  

    **Query Params:**
    - state_code: Short code of the state (Dropdown)
      -- example: DL, MH, TN, UP, AP, AR, AS, BR, CH, CG, GA, GJ, HR, HP, JK, JH, KA, KL, MN, ML, MZ, MP, NL, OD, PY, PB, RJ, SK, TS, TR, UK, WB
    - forecast_date: YYYY-MM-DD (optional, defaults to today)

    **Response 200 OK Example:**
    ```json
    {
        "state": "Delhi",
        "date": "2026-01-15",
        "average_load_mw": 3968.77,
        "peak_load_mw": 4696.96,
        "points": [
            {"datetime": "2026-01-15T00:00:00", "mw": 3530.95},
            {"datetime": "2026-01-15T00:05:00", "mw": 3540.12}
        ]
    }
    ```
    """
    #state_code  = state_code.value
    state_code = MERIT_TO_SHORT_MAP.get(state_code.value, state_code.value)
    start_date=query.forecast_date
    data = build_5min_forecast_response(state=state_code, forecast_date=start_date)
    return data




@router.get("/forecast-30day", response={200: dict})
def forecast_30day(request, state_code: StateShortEnum, from_date: date = Query(...)):
    """
    **URL:** GET /forecast-30day
    **Description:** 30-day-ahead daily load forecast using Facebook Prophet
    (free / open source). Returns per-day predicted peak & average load, a free
    weather temperature for each day, and a High-Risk-Day classification.

    **Query Params:**
    - state_code: Short code of the state (Dropdown)
      -- example: DL, CHG, MHA, UP, AP, ...
    - from_date: YYYY-MM-DD — first day of the 30-day window

    **Risk levels** (predicted daily peak vs historical peak):
    - NORMAL
    - HIGH RISK  (>= 85% of historical peak)
    - CRITICAL   (>= 95% of historical peak)

    **Weather** (free only): Open-Meteo forecast for the first ~16 days,
    Open-Meteo archive climatology (same calendar day averaged over the past
    3 years) for the remaining days.

    **Response 200 OK Example:**
    ```json
    {
        "state": "CG",
        "state_code": "CG",
        "model": "prophet",
        "from_date": "2026-07-01",
        "to_date": "2026-07-30",
        "days_forecasted": 30,
        "historical_peak_mw": 4525.0,
        "risk_thresholds_mw": {"high_risk": 3846.25, "critical": 4298.75},
        "risk_summary": {"normal_days": 25, "high_risk_days": 4, "critical_days": 1},
        "days": [
            {
                "date": "2026-07-01",
                "weekday": "Wednesday",
                "is_weekend": false,
                "is_holiday": false,
                "predicted_peak_load_mw": 3990.12,
                "predicted_average_load_mw": 3320.55,
                "peak_pct_of_historical": 88.2,
                "risk_level": "HIGH RISK",
                "temperature_c": 27.3,
                "weather_source": "climatology"
            }
        ]
    }
    ```
    """
    short = MERIT_TO_SHORT_MAP.get(state_code.value, state_code.value)
    return build_30day_forecast_response(state=short, from_date=from_date)







# @router.get("/temperature", response=TemperatureOut)
# def temperature_api(request, state_code: StateShortEnum, query: DateQuerySchema = Query(...)):
#     """
#     **URL:** GET /temperature  
#     **Description:** Returns hourly temperature and daily average temperature for a state. Fetches data automatically if missing.  

#     **Query Params:**
#     - state_code: Short code of the state (Dropdown)
#     -- example: DL, MH, TN, UP, AP, AR, AS, BR, CH, CG, GA, GJ, HR, HP, JK, JH, KA, KL, MN, ML, MZ, MP, NL, OD, PY, PB, RJ, SK, TS, TR, UK, WB
#     - forecast_date: YYYY-MM-DD (optional, defaults to today)

#     **Response 200 OK Example:**
#     ```json
#     {
#         "state": "WB",
#         "date": "2026-01-09",
#         "average_temperature": 22.5,
#         "hourly": [
#             {"time": "2026-01-09T00:00:00", "temp": 21.5}
#         ]
#     }
#     ```
#     """

#     forecast_date = query.forecast_date

#     qs = Weather.objects.filter(
#         state=state_code.value,
#         datetime__date=forecast_date
#     )

#     # 🔥 AUTO FETCH IF DATA NOT FOUND
#     if not qs.exists():
#         fetch_and_save_weather(
#             state_short=state_code.value,
#             start_date=forecast_date,
#             frequency="hourly"
#         )

#         qs = Weather.objects.filter(
#             state=state_code.value,
#             datetime__date=forecast_date
#         )

#     # STILL EMPTY → graceful response
#     if not qs.exists():
#         return {
#             "state": state_code.value,
#             "date": forecast_date.isoformat(),
#             "average_temperature": None,
#             "hourly": []
#         }

#     avg_temp = qs.aggregate(t=Avg("temperature_c"))["t"]

#     hourly = [
#         {
#             "time": obj.datetime.isoformat(),
#             "temp": round(obj.temperature_c, 1)
#         }
#         for obj in qs.order_by("datetime")
#     ]

#     return {
#         "state": state_code.value,
#         "date": forecast_date.isoformat(),
#         "average_temperature": round(avg_temp, 1),
#         "hourly": hourly
#     }




@router.get(
    "/previous-predictions",
    response=List[PreviousPredictionItem]
)
@paginate(PageNumberPagination, page_size=10)
def previous_predictions(
    request,
    state: StateShortEnum,
    date:DateQuerySchema = Query(...)
): 
    """
    **URL:** GET /previous-predictions  
    **Description:** Returns previously saved daily predictions. Supports pagination.  

    **Query Params:**
    - state_code: Short code of the state (Dropdown)
    -- example: DL, MH, TN, UP, AP, AR, AS, BR, CH, CG, GA, GJ, HR, HP, JK, JH, KA, KL, MN, ML, MZ, MP, NL, OD, PY, PB, RJ, SK, TS, TR, UK, WB
    - forecast_date: YYYY-MM-DD (optional, defaults to today)

    **Response 200 OK Example:**
    ```json
    [
        {"state": "WB", "date": "2026-01-09", "load_mw": 480.5},
        {"state": "WB", "date": "2026-01-08", "load_mw": 470.3}
    ]
    ```
    """

    qs = DailyPredictionHistory.objects.all()

    short_code = MERIT_TO_SHORT_MAP.get(state.value, state.value)

    if state:
        qs = qs.filter(state=short_code)

    if date:
        qs = qs.filter(date=date.forecast_date)

    return [
        {
            "state": obj.state,
            "date": obj.date.isoformat(),
            "load_mw": round(obj.load_mw, 2)
        }
        for obj in qs
    ]








# # 🚀 API Endpoint with schema
# @router.get("/state-current", response=List[MeritStateCurrentOut])
# def get_current_state_status(request, state: StateShortEnum):
#     """
#     *URL:* GET /state-current  
#     *Description:* Fetches current state-wise status from MERIT India website.  

#     *Query Params:*
#     - state: Short code of the state (e.g., DL, MH, TN)

#     *Response Example:*
#     json
#     {
#         "Demand": "3,598",
#         "ISGS": "366",
#         "ImportData": "3,232"
#     }
    
#     """
#     try:
#         url = f"https://meritindia.in/StateWiseDetails/BindCurrentStateStatus?StateCode={state.value}"
        
#         response = requests.get(url, timeout=10, verify=False)
#         response.raise_for_status()
        
#         data = response.json()
#         return data
        
#     except requests.Timeout:
#         raise HttpError(status_code=504, message="Request to MERIT India timed out")
#     except requests.HTTPError as e:
#         raise HttpError(status_code=e.response.status_code, message=f"Error from MERIT India: {e}")
#     except Exception as e:
#         raise HttpError(status_code=500, message=f"Error: {str(e)}")
    








@router.get("/state-current", response=List[MeritStateCurrentOut])
def get_current_state_status(request, state: StateShortEnum):

    try:
        merit_code = state.value

        # agar internal logic me use karna ho
        # state_enum = StateShortEnum(short_code)

        url = f"https://meritindia.in/StateWiseDetails/BindCurrentStateStatus?StateCode={merit_code}"

        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()

        return response.json()

    except Exception as e:
        raise HttpError(status_code=500, message=str(e))








    

