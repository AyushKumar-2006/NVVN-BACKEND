#!/usr/bin/env python3
"""
Standalone MERIT India CG demand collector — for GitHub Actions / cron.

Fetches Chhattisgarh live demand (MW) from MERIT and APPENDS one row to a CSV:

    timestamp_ist, demand_mw, state, isgs_mw, import_mw, national_mw, endpoint

No Django required — only `requests` plus the standard library — so it runs in a
minimal CI environment with `pip install requests`.

    python scripts/collect_merit_data.py --csv data/cg_live_data.csv

Notes
-----
* The verified endpoint is /StateWiseDetails/BindCurrentStateStatus; the
  /Dashboard/... path (often quoted) returns an HTML error, so we try the
  Dashboard path first then fall back to the working one.
* MERIT's feed is a LIVE snapshot and is frequently idle (Demand=null). A null
  reading is still written (demand_mw left blank) so the polling cadence stays
  visible in the CSV; pass --skip-null to omit those rows instead.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests

try:
    requests.packages.urllib3.disable_warnings()
except Exception:  # noqa: BLE001
    pass

IST = timezone(timedelta(hours=5, minutes=30))

STATE_STATUS_ENDPOINTS = [
    "https://meritindia.in/Dashboard/BindCurrentStateStatus",
    "https://meritindia.in/StateWiseDetails/BindCurrentStateStatus",
]
ALL_INDIA_URL = "https://meritindia.in/Dashboard/BindAllIndiaMap"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 MERIT-collect",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://meritindia.in/",
}
CSV_HEADER = ["timestamp_ist", "demand_mw", "state", "isgs_mw",
              "import_mw", "national_mw", "endpoint"]


def _to_float(value):
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def fetch_state(state_name, timeout):
    """Return (demand, isgs, import_, endpoint). demand is None if idle/failed."""
    for url in STATE_STATUS_ENDPOINTS:
        try:
            r = requests.get(url, params={"StateName": state_name},
                             headers=HEADERS, timeout=timeout, verify=False)
        except requests.exceptions.RequestException:
            continue
        if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
            continue
        try:
            data = r.json()
        except ValueError:
            continue
        row = (data[0] if isinstance(data, list) and data
               else data if isinstance(data, dict) else {})
        return (_to_float(row.get("Demand")), _to_float(row.get("ISGS")),
                _to_float(row.get("ImportData")), url)
    return None, None, None, None


def fetch_national(timeout):
    try:
        r = requests.get(ALL_INDIA_URL, headers=HEADERS, timeout=timeout, verify=False)
        m = re.search(r"DEMAND MET[^0-9]*([\d,]+)", r.text)
        return float(m.group(1).replace(",", "")) if m else None
    except (requests.exceptions.RequestException, ValueError):
        return None


def append_row(path, row, skip_null):
    if skip_null and row[1] in (None, ""):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(CSV_HEADER)
        w.writerow(["" if v is None else v for v in row])
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description="Append a MERIT CG demand reading to CSV.")
    ap.add_argument("--csv", default="data/cg_live_data.csv", help="output CSV path")
    ap.add_argument("--state", default="Chhattisgarh", help="MERIT state name")
    ap.add_argument("--state-code", default="CG", help="state code written to CSV")
    ap.add_argument("--timeout", type=int, default=20, help="request timeout (s)")
    ap.add_argument("--skip-null", action="store_true",
                    help="do not write rows when demand is null/idle")
    args = ap.parse_args(argv)

    ts = datetime.now(IST).replace(microsecond=0).isoformat()
    demand, isgs, imp, endpoint = fetch_state(args.state, args.timeout)
    national = fetch_national(args.timeout)

    row = [ts, demand, args.state_code, isgs, imp, national, endpoint or ""]
    written = append_row(args.csv, row, args.skip_null)

    status = "idle/null" if demand is None else f"{demand:,.0f} MW"
    print(f"{ts}  CG demand={status}  national={national}  "
          f"{'written' if written else 'skipped'} -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
