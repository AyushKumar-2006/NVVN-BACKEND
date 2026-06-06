#!/usr/bin/env python
"""
wrldc_fetcher.py  —  Ingest real INTRADAY load for Chhattisgarh from WRLDC
                     (Western Regional Load Despatch Centre), 15-min -> 5-min.

What it does
------------
- Gets a 15-minute CG load series (96 blocks/day) — either from a WRLDC
  data-dashboard endpoint, or from a file you downloaded by hand.
- Interpolates 15-min -> 5-min (288 points/day) with pandas.
- Saves to the existing `StateLoad5Min` table by reusing the project's existing
  upload pipeline (`power.utils.upload.save_state_5min_generic`) with the
  standard `DateTime / State / Load_MW` schema.

Why
---
Gives real recent intraday data for the XGBoost 5-minute model instead of the
synthetic series.

IMPORTANT — this only ADDS a new file. It does not modify any existing code; it
reuses existing helpers read-only.

Honest caveats (please read)
----------------------------
1. WRLDC's data dashboard is a JavaScript app; its backend JSON endpoint is not
   publicly documented and changes. `WRLDC_ENDPOINT` / `parse_wrldc_payload()`
   below are a best-effort template — VERIFY them, or just use `--csv`/`--xlsx`
   mode on an export you downloaded from the dashboard. The file mode works today
   and exercises the full resample + save pipeline.
2. The file reader auto-detects common column names; pass `--datetime-col` /
   `--load-col` if your export uses different headers.
3. Saving overwrites any existing CG rows in the same datetime range (the
   existing pipeline does a clean range-replace) — expected for a re-import.

Usage
-----
    # from a manually downloaded WRLDC export (CSV or XLSX) -- works now
    python wrldc_fetcher.py --csv ~/Downloads/wrldc_cg_2025-05.csv
    python wrldc_fetcher.py --xlsx ~/Downloads/wrldc_cg.xlsx \
        --datetime-col "Time" --load-col "CG"

    # try the live endpoint for a date range (verify WRLDC_ENDPOINT first)
    python wrldc_fetcher.py --from 2025-05-01 --to 2025-05-07

    # transform only, do not write to the DB
    python wrldc_fetcher.py --csv export.csv --dry-run
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Django bootstrap (script lives at the project root)
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django  # noqa: E402

django.setup()

from power.models import StateLoad5Min  # noqa: E402
from power.utils.upload import save_state_5min_generic  # noqa: E402

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
STATE_CODE = "CG"

# Verified WRLDC real-time endpoint (the one its own chart.html / data dashboard
# calls). It is an ASP.NET PageMethod: POST a JSON body, the result is wrapped in
# a top-level {"d": "<json-string>"}.
#   - request body : {date:"YYYY-M-D"}   (month/day are NOT zero-padded)
#   - state name   : WRLDC spells Chhattisgarh "Chattishgarh"
#   - load value   : the "Demand" field (state demand met, MW)
#   - timestamp    : "current_datetime", in %Y-%d-%m %H:%M:%S (year-DAY-month!)
# NOTE: this is a *real-time* feed — it serves the current day's accumulating
# 15-min blocks, not historical days. For history, export from the dashboard and
# use --csv/--xlsx. The GetAllScheduleDates method reports which date is live.
WRLDC_ENDPOINT = "https://wrldc.in/OnlinestateTest1.aspx/GetRealTimeData_state_Wise"
WRLDC_DATES_ENDPOINT = "https://wrldc.in/OnlinestateTest1.aspx/GetAllScheduleDates"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (ingestion script; +local)",
    "Content-Type": "application/json; charset=utf-8",
    "X-Requested-With": "XMLHttpRequest",
}

# WRLDC's spelling(s) of Chhattisgarh and the field we treat as the load.
CG_NAME_MATCH = "chat"          # matches "Chattishgarh" (and "Chhattisgarh")
LOAD_FIELD = "Demand"            # state demand met (MW)
WRLDC_DT_FMT = "%Y-%d-%m %H:%M:%S"   # e.g. "2025-10-05 03:15:05" -> 10 May 2025

# Column-name candidates when reading a downloaded export
DATETIME_CANDIDATES = ["datetime", "date_time", "timestamp", "time", "date", "block_time"]
LOAD_CANDIDATES = ["cg", "chhattisgarh", "load_mw", "load", "actual", "drawal", "mw", "value"]


# --------------------------------------------------------------------------
# Source A: local export (works today)
# --------------------------------------------------------------------------
def _find_col(cols, candidates, override=None):
    if override:
        for c in cols:
            if c.strip().lower() == override.strip().lower():
                return c
        raise ValueError(f"column {override!r} not found in {list(cols)}")
    low = {c.strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    # fuzzy contains
    for cand in candidates:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def load_local(path: str, dt_col=None, load_col=None) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    dcol = _find_col(raw.columns, DATETIME_CANDIDATES, dt_col)
    lcol = _find_col(raw.columns, LOAD_CANDIDATES, load_col)
    if not dcol or not lcol:
        raise ValueError(
            f"could not detect datetime/load columns in {list(raw.columns)}; "
            f"pass --datetime-col and --load-col"
        )

    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw[dcol], errors="coerce", dayfirst=True),
        "load_mw": pd.to_numeric(raw[lcol], errors="coerce"),
    }).dropna()
    print(f"  read {len(df)} rows from {os.path.basename(path)} "
          f"(datetime='{dcol}', load='{lcol}')")
    return df


# --------------------------------------------------------------------------
# Source B: live WRLDC endpoint (best-effort; verify before relying on it)
# --------------------------------------------------------------------------
def _unwrap_aspnet(resp_json):
    """ASP.NET PageMethods return {"d": <payload-or-json-string>}."""
    inner = resp_json.get("d", resp_json) if isinstance(resp_json, dict) else resp_json
    if isinstance(inner, str):
        inner = json.loads(inner)
    return inner


def parse_wrldc_payload(payload, target_date: date) -> pd.DataFrame:
    """
    Map a WRLDC state-wise payload to [datetime, load_mw] for Chhattisgarh.

    Each item looks like:
        {"StateName": "Chattishgarh", "Demand": "3421.0",
         "current_datetime": "2025-10-05 03:15:05", ...}
    Zero / non-positive demand rows are dropped (the feed pads missing blocks
    with 0.0, which would otherwise poison the load series).
    """
    rows = payload if isinstance(payload, list) else (payload or {}).get("data", [])
    out = []
    for item in rows or []:
        if not isinstance(item, dict):
            continue
        if CG_NAME_MATCH not in str(item.get("StateName", "")).lower():
            continue
        t = item.get("current_datetime")
        v = item.get(LOAD_FIELD)
        if t is None or v is None:
            continue
        try:
            ts = datetime.strptime(str(t), WRLDC_DT_FMT)
            load = float(v)
        except (ValueError, TypeError):
            continue
        if load <= 0:          # skip padded / not-yet-reported blocks
            continue
        out.append({"datetime": ts, "load_mw": load})
    return pd.DataFrame(out)


def _wrldc_date_str(d: date) -> str:
    """WRLDC wants a non-zero-padded date: '2025-5-1', not '2025-05-01'."""
    return f"{d.year}-{d.month}-{d.day}"


def get_available_dates() -> list[date]:
    """Dates the live feed currently has data for (usually just the live day)."""
    try:
        r = requests.post(WRLDC_DATES_ENDPOINT, data='{flag:"1"}',
                          headers=HEADERS, timeout=30, verify=False)
        r.raise_for_status()
        raw = _unwrap_aspnet(r.json()) or []
        out = []
        for s in raw:
            for fmt in ("%m/%d/%Y", "%Y-%d-%m %H:%M:%S", "%Y-%m-%d"):
                try:
                    out.append(datetime.strptime(str(s), fmt).date())
                    break
                except ValueError:
                    continue
        return sorted(set(out))
    except Exception as e:  # noqa: BLE001
        print(f"  !! could not list WRLDC dates: {e}")
        return []


def fetch_live(target_date: date) -> pd.DataFrame:
    body = '{date:"' + _wrldc_date_str(target_date) + '"}'
    try:
        r = requests.post(WRLDC_ENDPOINT, data=body, headers=HEADERS,
                          timeout=40, verify=False)
        r.raise_for_status()
        return parse_wrldc_payload(_unwrap_aspnet(r.json()), target_date)
    except Exception as e:  # noqa: BLE001
        print(f"  !! WRLDC fetch failed for {target_date}: {e}")
        print(f"     the feed may be down — export from the dashboard and use --csv/--xlsx")
        return pd.DataFrame()


# --------------------------------------------------------------------------
# Transform: 15-min -> 5-min
# --------------------------------------------------------------------------
def resample_15min_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    s = (
        df.dropna(subset=["datetime"])
        .drop_duplicates(subset=["datetime"])
        .set_index("datetime")
        .sort_index()["load_mw"]
    )
    five = s.resample("5min").interpolate("time")
    out = five.reset_index()
    out.columns = ["datetime", "load_mw"]
    return out


def to_upload_frame(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Build the generic `DateTime / State / Load_MW` schema the pipeline expects."""
    return pd.DataFrame({
        "DateTime": df_5min["datetime"],
        "State": STATE_CODE,
        "Load_MW": df_5min["load_mw"].round(2),
    })


def save(df_5min: pd.DataFrame, dry_run: bool = False) -> int:
    if df_5min.empty:
        print("  nothing to save")
        return 0
    upload_df = to_upload_frame(df_5min)
    if dry_run:
        print(f"  [dry-run] would upsert {len(upload_df)} StateLoad5Min rows "
              f"({upload_df['DateTime'].min()} .. {upload_df['DateTime'].max()})")
        return 0
    # reuse the existing upload pipeline (delete-range + bulk insert)
    return save_state_5min_generic(upload_df)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="WRLDC 15-min -> 5-min -> StateLoad5Min (CG)")
    ap.add_argument("--csv", help="path to a downloaded WRLDC CSV export")
    ap.add_argument("--xlsx", help="path to a downloaded WRLDC XLSX export")
    ap.add_argument("--datetime-col", help="datetime column name in the export")
    ap.add_argument("--load-col", help="CG load column name in the export")
    ap.add_argument("--from", dest="dfrom", help="live fetch range start YYYY-MM-DD")
    ap.add_argument("--to", dest="dto", help="live fetch range end YYYY-MM-DD")
    ap.add_argument("--latest", action="store_true",
                    help="fetch whatever date the live feed currently has (default "
                         "if no other source is given)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.csv or args.xlsx:
        raw15 = load_local(args.csv or args.xlsx, args.datetime_col, args.load_col)
    elif args.dfrom and args.dto:
        cur, end = date.fromisoformat(args.dfrom), date.fromisoformat(args.dto)
        frames = []
        while cur <= end:
            print(f"Fetching WRLDC CG for {cur}")
            frames.append(fetch_live(cur))
            cur += timedelta(days=1)
        raw15 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        # default / --latest: pull the live feed's current date(s)
        avail = get_available_dates()
        if not avail:
            print("Live feed reports no available dates. Export from the dashboard "
                  "and re-run with --csv/--xlsx.")
            return
        print(f"Live feed available date(s): {[d.isoformat() for d in avail]}")
        frames = [fetch_live(d) for d in avail]
        raw15 = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if raw15.empty:
        print("No 15-min data obtained — nothing to do.")
        return

    print(f"Got {len(raw15)} 15-min points "
          f"({raw15['datetime'].min()} .. {raw15['datetime'].max()})")
    five = resample_15min_to_5min(raw15)
    print(f"Resampled to {len(five)} 5-min points")

    saved = save(five, dry_run=args.dry_run)
    print(f"\nSaved {saved} row(s) to StateLoad5Min (CG) via the existing pipeline.")


if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()
    main()
