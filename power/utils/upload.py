import pandas as pd
import pytz
from django.db import transaction
from django.db.models import Q
from django.core.exceptions import ValidationError
from power.models import (RegionHourlyLoad,StateDailyLoad,StateLoad5Min)



# =====================================================
# CONSTANTS
# =====================================================

IST = pytz.timezone("Asia/Kolkata")
SQLITE_BATCH_SIZE = 400   # safe for sqlite

REGION_COLUMN_MAP = {
    "NR": "Northen Region Hourly Demand",
    "WR": "Western Region Hourly Demand",
    "ER": "Eastern Region Hourly Demand",
    "SR": "Southern Region Hourly Demand",
    "NER": "North-Eastern Region Hourly Demand",
}

STATE_SHORT_MAP = {
    "Andhra Pradesh": "AP",
    "Arunachal Pradesh": "AR",
    "Assam": "AS",
    "Bihar": "BR",
    "Chandigarh": "CH",
    "Chhattisgarh": "CG",
    "Delhi": "DL",
    "Goa": "GA",
    "Gujarat": "GJ",
    "Haryana": "HR",
    "Himachal Pradesh": "HP",
    "Jammu & Kashmir": "JK",
    "Jharkhand": "JH",
    "Karnataka": "KA",
    "Kerala": "KL",
    "Maharashtra": "MH",
    "Manipur": "MN",
    "Meghalaya": "ML",
    "Mizoram": "MZ",
    "Madhya Pradesh": "MP",
    "Nagaland": "NL",
    "Odisha": "OD",
    "Puducherry": "PY",
    "Punjab": "PB",
    "Rajasthan": "RJ",
    "Sikkim": "SK",
    "Tamil Nadu": "TN",
    "Telangana": "TS",
    "Tripura": "TR",
    "Uttar Pradesh": "UP",
    "Uttarakhand": "UK",
    "West Bengal": "WB",
}

SHORT_CODES = set(STATE_SHORT_MAP.values())
SKIP_COLUMNS = {"Dates", "Total Consumption", "Unnamed: 0"}









# =====================================================
# HELPERS
# =====================================================
def normalize_state(value):
    if not value:
        return None
    value = str(value).strip()
    if value in SHORT_CODES:
        return value
    return STATE_SHORT_MAP.get(value)








# =====================================================
# DUPLICATE CLEAN (CRITICAL FIX)
# =====================================================

def delete_state_5min_range(state, min_dt, max_dt):
    StateLoad5Min.objects.filter(
        state=state,
        datetime__gte=min_dt,
        datetime__lte=max_dt
    ).delete()






# =====================================================
# BULK UPSERT HELPERS (PAIR-WISE SAFE)
# =====================================================

def bulk_upsert_region(records):
    existing = {}

    for i in range(0, len(records), SQLITE_BATCH_SIZE):
        chunk = records[i:i + SQLITE_BATCH_SIZE]
        q = Q()
        for r in chunk:
            q |= Q(region=r.region, datetime=r.datetime)

        for obj in RegionHourlyLoad.objects.filter(q):
            existing[(obj.region, obj.datetime)] = obj

    to_create, to_update = [], []

    for r in records:
        key = (r.region, r.datetime)
        if key in existing:
            obj = existing[key]
            obj.load_mw = r.load_mw
            to_update.append(obj)
        else:
            to_create.append(r)

    with transaction.atomic():
        if to_create:
            RegionHourlyLoad.objects.bulk_create(to_create, batch_size=1000)
        if to_update:
            RegionHourlyLoad.objects.bulk_update(
                to_update,
                ["load_mw"],
                batch_size=1000
            )







def bulk_upsert_state_daily(records):
    existing = {}

    for i in range(0, len(records), SQLITE_BATCH_SIZE):
        chunk = records[i:i + SQLITE_BATCH_SIZE]
        q = Q()
        for r in chunk:
            q |= Q(state=r.state, date=r.date)

        for obj in StateDailyLoad.objects.filter(q):
            existing[(obj.state, obj.date)] = obj

    to_create, to_update = [], []

    for r in records:
        key = (r.state, r.date)
        if key in existing:
            obj = existing[key]
            obj.energy_mu = r.energy_mu
            to_update.append(obj)
        else:
            to_create.append(r)

    with transaction.atomic():
        if to_create:
            StateDailyLoad.objects.bulk_create(to_create, batch_size=2000)
        if to_update:
            StateDailyLoad.objects.bulk_update(
                to_update,
                ["energy_mu"],
                batch_size=2000
            )
        





def bulk_upsert_state_5min(records):
    if not records:
        return

    existing = {}

    # ---- chunked fetch ----
    for i in range(0, len(records), SQLITE_BATCH_SIZE):
        chunk = records[i:i + SQLITE_BATCH_SIZE]
        q = Q()
        for r in chunk:
            q |= Q(state=r.state, datetime=r.datetime)

        for obj in StateLoad5Min.objects.filter(q):
            existing[(obj.state, obj.datetime)] = obj

    to_create = []
    to_update = []

    for r in records:
        key = (r.state, r.datetime)
        if key in existing:
            obj = existing[key]
            obj.load_mw = r.load_mw
            obj.brpl = r.brpl
            obj.bypl = r.bypl
            obj.ndpl = r.ndpl
            obj.ndmc = r.ndmc
            obj.mes = r.mes
            to_update.append(obj)
        else:
            to_create.append(r)

    with transaction.atomic():
        if to_create:
            StateLoad5Min.objects.bulk_create(
                to_create,
                batch_size=SQLITE_BATCH_SIZE
            )

        if to_update:
            StateLoad5Min.objects.bulk_update(
                to_update,
                ["load_mw", "brpl", "bypl", "ndpl", "ndmc", "mes"],
                batch_size=SQLITE_BATCH_SIZE
            )







# =====================================================
# SAVE FUNCTIONS
# =====================================================

def save_region_hourly_load_from_xlsx(df):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    records = []

    for _, row in df.iterrows():
        if pd.isna(row["datetime"]):
            continue

        dt = row["datetime"]

        for region, col in REGION_COLUMN_MAP.items():
            if col not in df.columns or pd.isna(row[col]):
                continue

            records.append(
                RegionHourlyLoad(
                    region=region,
                    datetime=dt,
                    load_mw=float(row[col])
                )
            )

    if not records:
        raise ValidationError("No region hourly data found")

    bulk_upsert_region(records)
    return len(records)







def save_state_daily_load_from_csv(df):
    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce").dt.date
    records = []

    for _, row in df.iterrows():
        if pd.isna(row["Dates"]):
            continue

        for col in df.columns:
            if col in SKIP_COLUMNS:
                continue

            state = normalize_state(col)
            if not state or pd.isna(row[col]):
                continue

            records.append(
                StateDailyLoad(
                    state=state,
                    date=row["Dates"],
                    energy_mu=float(row[col])
                )
            )

    if not records:
        raise ValidationError("No state daily data found")

    bulk_upsert_state_daily(records)
    return len(records)





# def save_state_5min_load_from_csv(df):
#     required = ["DateTime", "Delhi", "BRPL", "BYPL", "NDPL", "NDMC", "MES"]
#     missing = [c for c in required if c not in df.columns]
#     if missing:
#         raise ValidationError(f"Missing columns: {missing}")

#     # ---- STRICT datetime parse (NAIVE IST) ----
#     df["DateTime"] = pd.to_datetime(
#         df["DateTime"],
#         dayfirst=True,
#         errors="raise"
#     )

#     # ---- Remove duplicate 5-min slots in CSV ----
#     df = df.drop_duplicates(subset=["DateTime"], keep="last")

#     # -------- State detection --------
#     non_datetime_cols = [c for c in df.columns if c != "DateTime"]

#     if len(non_datetime_cols) < 2:
#         raise ValidationError("CSV must contain State and Area columns")

#     raw_state_col = non_datetime_cols[0]
#     state = normalize_state(raw_state_col)

#     # ---- SAFE RANGE DELETE ----
#     min_dt = df["DateTime"].min()
#     max_dt = df["DateTime"].max()

#     delete_state_5min_range(
#         state=state,
#         min_dt=min_dt,
#         max_dt=max_dt
#     )

#     # ---- INSERT EXACT CSV VALUES ----
#     records = []
#     for _, row in df.iterrows():
#         records.append(
#             StateLoad5Min(
#                 state=state,
#                 datetime=row["DateTime"],
#                 load_mw=float(row["Delhi"]),
#                 brpl=float(row["BRPL"]),
#                 bypl=float(row["BYPL"]),
#                 ndpl=float(row["NDPL"]),
#                 ndmc=float(row["NDMC"]),
#                 mes=float(row["MES"]),
#             )
#         )

#     if not records:
#         raise ValidationError("No valid 5-minute data found")

#     StateLoad5Min.objects.bulk_create(
#         records,
#         batch_size=SQLITE_BATCH_SIZE
#     )

#     return len(records)




def save_state_5min_generic(df, cols=None):
    """
    Generic 5-minute state load format (CSV or XLSX):
        DateTime (or Date + Time), State, Load_MW

    State column may hold a short code (e.g. "CG") or full name
    (e.g. "Chhattisgarh"); both are normalised. Multiple states in one
    file are supported. Re-uploading the same range overwrites it.
    """
    if cols is None:
        cols = {c.strip().lower(): c for c in df.columns}

    state_col = cols["state"]
    load_col = cols["load_mw"]

    # ---- datetime (prefer explicit DateTime, else Date + Time) ----
    if "datetime" in cols:
        df["dtv"] = pd.to_datetime(df[cols["datetime"]], errors="coerce")
    else:
        df["dtv"] = pd.to_datetime(
            df[cols["date"]].astype(str) + " " + df[cols["time"]].astype(str),
            errors="coerce",
        )

    df["loadv"] = pd.to_numeric(df[load_col], errors="coerce")
    df["sv"] = df[state_col].map(normalize_state)

    df = df.dropna(subset=["dtv", "loadv", "sv"])
    df = df.drop_duplicates(subset=["sv", "dtv"], keep="last")

    if df.empty:
        raise ValidationError("No valid 5-minute state data found")

    # ---- idempotent overwrite: clear each state's datetime range ----
    for state, g in df.groupby("sv"):
        delete_state_5min_range(state, g["dtv"].min(), g["dtv"].max())

    records = [
        StateLoad5Min(state=s, datetime=d, load_mw=float(l))
        for s, d, l in zip(df["sv"], df["dtv"], df["loadv"])
    ]

    StateLoad5Min.objects.bulk_create(records, batch_size=SQLITE_BATCH_SIZE)
    return len(records)




def save_state_5min_load_from_csv(df):
    required = ["DateTime", "Delhi", "BRPL", "BYPL", "NDPL", "NDMC", "MES"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing columns: {missing}")

    # ---- strict datetime parse (IST-naive) ----
    df["DateTime"] = pd.to_datetime(
        df["DateTime"],
        dayfirst=True,
        errors="raise"
    )

    # ---- remove duplicate 5-min slots in CSV ----
    df = df.drop_duplicates(subset=["DateTime"], keep="last")

    # --------------------------------------------------
    # 🔥 STATE DETECTION (NO HARDCODE, NO HELPER)
    # --------------------------------------------------
    non_datetime_cols = [c for c in df.columns if c != "DateTime"]

    if not non_datetime_cols:
        raise ValidationError("CSV must contain state column")

    raw_state_col = non_datetime_cols[0]
    state = normalize_state(raw_state_col)

    if not state:
        raise ValidationError(f"Invalid state column: {raw_state_col}")

    # --------------------------------------------------
    # INSERT / UPSERT
    # --------------------------------------------------
    records = []
    for _, row in df.iterrows():
        records.append(
            StateLoad5Min(
                state=state,
                datetime=row["DateTime"],
                load_mw=float(row["Delhi"]),
                brpl=float(row["BRPL"]),
                bypl=float(row["BYPL"]),
                ndpl=float(row["NDPL"]),
                ndmc=float(row["NDMC"]),
                mes=float(row["MES"]),
            )
        )

    if not records:
        raise ValidationError("No valid 5-minute data found")

    bulk_upsert_state_5min(records)
    return len(records)









# =====================================================
# MAIN UPLOAD ENTRY
# =====================================================

def save_power_data_from_xlsx(file) -> int:
    filename = file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(file)
        file_type = "CSV"
    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        file_type = "XLSX"
    else:
        raise ValidationError("Unsupported file type")

    if df.empty:
        raise ValidationError("Uploaded file is empty")

    # ---- generic 5-min state load: DateTime/Date+Time + State + Load_MW ----
    cols = {c.strip().lower(): c for c in df.columns}
    if (
        "state" in cols
        and "load_mw" in cols
        and ("datetime" in cols or ("date" in cols and "time" in cols))
    ):
        return save_state_5min_generic(df, cols)

    if file_type == "CSV" and "DateTime" in df.columns:
        return save_state_5min_load_from_csv(df)

    if file_type == "CSV" and "Dates" in df.columns:
        return save_state_daily_load_from_csv(df)

    if file_type == "XLSX" and "datetime" in df.columns:
        return save_region_hourly_load_from_xlsx(df)

    raise ValidationError("Unsupported file format / columns")
