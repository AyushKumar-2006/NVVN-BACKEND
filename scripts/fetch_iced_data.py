#!/usr/bin/env python3
"""
Auto-download Chhattisgarh hourly load-curve data from NITI Aayog's ICED portal.

The ICED site (https://iced.niti.gov.in/.../load-curve) is an Angular SPA whose
"Download XLS" button builds the workbook *client-side* from a JSON API — so the
JSON endpoint below IS the data source the XLS is made from.

    Base : https://icedapi.niti.gov.in/v1
    Data : /energy/electricity/distribution/loadCurveHourlyState
           ?year=<y1,y2>&state=Chhattisgarh&demand=Maximum
    Years: /energy/electricity/distribution/loadCurveFilters  -> {"year":[...], ...}

The hourly-state response is a JSON array, one object per requested year; each
object has a leading "year" key and one key per *state name* whose value is a list
of points {hour, demandMet, date}. We flatten Chhattisgarh's points to CSV rows:

    fetched_at_ist, year, date, state, hour, demand_met, demand_type, endpoint

Only `requests` + the stdlib are needed (mirrors scripts/collect_merit_data.py),
so it runs in a minimal CI container with `pip install requests`.

    python scripts/fetch_iced_data.py --csv data/cg_hourly_iced.csv

By default rows are *upserted* keyed on (state, demand_type, year, hour): the
hourly load curve is a yearly aggregate, so a plain daily append would just pile
up duplicate rows. Upsert keeps the CSV idempotent — re-running refreshes the
current year in place and only adds genuinely new (year/hour) rows. Pass
--append for raw append-only behaviour instead.

NOTE: icedapi.niti.gov.in (NIC / gov.in) is geo-restricted and is unreachable
from many networks outside India; run this from an India host or GitHub Actions.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

try:  # gov certs occasionally present an incomplete chain
    requests.packages.urllib3.disable_warnings()
except Exception:  # noqa: BLE001
    pass

IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_BASE = os.environ.get("ICED_API_BASE", "https://icedapi.niti.gov.in/v1")
FILTERS_PATH = "/energy/electricity/distribution/loadCurveFilters"
HOURLY_STATE_PATH = "/energy/electricity/distribution/loadCurveHourlyState"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 ICED-collect",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://iced.niti.gov.in",
    "Referer": "https://iced.niti.gov.in/energy/electricity/distribution/"
               "national-level-consumption/load-curve",
    "X-Requested-With": "XMLHttpRequest",
}

CSV_HEADER = ["fetched_at_ist", "year", "date", "state",
              "hour", "demand_met", "demand_type", "endpoint"]
# columns that identify a unique observation (for --upsert dedup)
KEY_COLS = ("state", "demand_type", "year", "hour")

# keys that appear alongside the per-state series and must not be treated as states
NON_STATE_KEYS = {"year", "y", "d", "date", "region"}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def _get_json(session, url, params, timeout):
    """GET url -> parsed JSON. Retries once with verify=False on SSL trouble."""
    for verify in (True, False):
        try:
            r = session.get(url, params=params, headers=HEADERS,
                            timeout=timeout, verify=verify)
        except requests.exceptions.SSLError:
            continue  # retry without verification
        except requests.exceptions.RequestException as exc:
            raise SystemExit(f"ERROR: request to {url} failed: {exc}")
        if r.status_code != 200:
            raise SystemExit(f"ERROR: {url} -> HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()
        except ValueError:
            raise SystemExit(f"ERROR: {url} did not return JSON: {r.text[:200]}")
    raise SystemExit(f"ERROR: SSL verification failed for {url}")


def discover_years(session, base, timeout):
    """Return the sorted list of available years from loadCurveFilters, or []."""
    data = _get_json(session, base + FILTERS_PATH, None, timeout)
    years = data.get("year") if isinstance(data, dict) else None
    return [str(y) for y in years] if isinstance(years, list) else []


def fetch_hourly_state(session, base, years, states, demand, timeout):
    """GET loadCurveHourlyState for the given comma-joined year/state strings."""
    params = {"year": ",".join(years), "state": ",".join(states), "demand": demand}
    return _get_json(session, base + HOURLY_STATE_PATH, params, timeout)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _norm(s):
    return str(s).strip().lower()


def _to_float(value):
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def parse_hourly_state(payload, target_state, demand_type, fetched_at):
    """Flatten the loadCurveHourlyState array into CSV rows for `target_state`.

    Tolerant of shape drift: each list item is a per-year dict; any key whose
    value is a list of hourly points is treated as a state series, and we keep
    the one matching `target_state` (case-insensitive).
    """
    if not isinstance(payload, list):
        raise SystemExit(f"ERROR: expected a JSON array, got {type(payload).__name__}")

    target = _norm(target_state)
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        item_year = item.get("year") or item.get("y")
        for key, series in item.items():
            if _norm(key) in NON_STATE_KEYS or not isinstance(series, list):
                continue
            if _norm(key) != target:
                continue
            for point in series:
                if not isinstance(point, dict):
                    continue
                date = point.get("date") or point.get("d") or ""
                # year is most reliably taken from the point's own date
                year = (str(date)[:4] if str(date)[:4].isdigit()
                        else (str(item_year) if item_year else ""))
                rows.append({
                    "fetched_at_ist": fetched_at,
                    "year": year,
                    "date": date,
                    "state": key,
                    "hour": point.get("hour", ""),
                    "demand_met": _to_float(point.get("demandMet")),
                    "demand_type": demand_type,
                    "endpoint": HOURLY_STATE_PATH,
                })
    return rows


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def _read_existing(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k))
                        for k in CSV_HEADER})


def _key(row):
    return tuple(str(row.get(c, "")) for c in KEY_COLS)


def save_rows(path, new_rows, append):
    """Write rows to CSV. Returns (n_added, n_updated, n_total)."""
    if append:
        existing = _read_existing(path)
        added = len(new_rows)
        _write_all(path, existing + new_rows)
        return added, 0, len(existing) + added

    # upsert keyed on KEY_COLS
    merged = {_key(r): r for r in _read_existing(path)}
    added = updated = 0
    for row in new_rows:
        k = _key(row)
        if k in merged:
            if any(str(merged[k].get(c, "")) != str(row.get(c) if row.get(c) is not None else "")
                   for c in ("demand_met", "date")):
                updated += 1
            merged[k] = row
        else:
            added += 1
            merged[k] = row
    ordered = sorted(merged.values(),
                     key=lambda r: (str(r.get("state")), str(r.get("demand_type")),
                                    str(r.get("year")), str(r.get("hour"))))
    _write_all(path, ordered)
    return added, updated, len(ordered)


# --------------------------------------------------------------------------- #
# Self-test (offline parser check — no network)
# --------------------------------------------------------------------------- #
def _self_test():
    sample = [
        {"year": "2024",
         "Chhattisgarh": [
             {"hour": "00:00", "demandMet": 3500, "date": "2024-06-01T00:00:00Z"},
             {"hour": "01:00", "demandMet": 3420, "date": "2024-06-01T00:00:00Z"}],
         "Madhya Pradesh": [
             {"hour": "00:00", "demandMet": 9000, "date": "2024-06-01T00:00:00Z"}]},
        {"year": "2025",
         "Chhattisgarh": [
             {"hour": "00:00", "demandMet": 3700, "date": "2025-06-01T00:00:00Z"}]},
    ]
    rows = parse_hourly_state(sample, "Chhattisgarh", "Maximum", "TS")
    assert len(rows) == 3, rows
    assert {r["year"] for r in rows} == {"2024", "2025"}, rows
    assert all(r["state"] == "Chhattisgarh" for r in rows), rows
    assert rows[0]["demand_met"] == 3500.0, rows[0]

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        a, u, t = save_rows(p, rows, append=False)
        assert (a, u, t) == (3, 0, 3), (a, u, t)
        # re-run with one changed value -> 1 update, 0 add
        rows2 = [dict(r) for r in rows]
        rows2[0]["demand_met"] = 3600.0
        a, u, t = save_rows(p, rows2, append=False)
        assert (a, u, t) == (0, 1, 3), (a, u, t)
    print("self-test OK")
    return 0


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/cg_hourly_iced.csv", help="output CSV path")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="ICED API base URL")
    ap.add_argument("--state", default="Chhattisgarh", help="state name as ICED spells it")
    ap.add_argument("--extra-states", default="Madhya Pradesh",
                    help="comma list sent with --state to satisfy ICED's >=2-state "
                         "filter; only --state rows are kept (set empty to send one)")
    ap.add_argument("--demand", default="Maximum",
                    choices=["Maximum", "Minimum", "Average"],
                    help="demand statistic for the hourly curve")
    ap.add_argument("--years", default="",
                    help="comma list of years; default = latest available year")
    ap.add_argument("--all-years", action="store_true",
                    help="fetch every year reported by loadCurveFilters")
    ap.add_argument("--append", action="store_true",
                    help="append rows instead of upserting on (state,demand,year,hour)")
    ap.add_argument("--timeout", type=int, default=30, help="request timeout (s)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the offline parser self-test and exit (no network)")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    base = args.base_url.rstrip("/")
    fetched_at = datetime.now(IST).replace(microsecond=0).isoformat()
    session = requests.Session()

    # 1. determine which years to fetch
    if args.years.strip():
        years = [y.strip() for y in args.years.split(",") if y.strip()]
    else:
        available = discover_years(session, base, args.timeout)
        if not available:
            raise SystemExit("ERROR: could not discover years from loadCurveFilters; "
                             "pass --years explicitly.")
        years = available if args.all_years else [available[-1]]
    print(f"Years: {','.join(years)}  state={args.state}  demand={args.demand}")

    # 2. build the state list (target + fillers to satisfy the >=2 filter)
    states = [args.state] + [s.strip() for s in args.extra_states.split(",") if s.strip()]
    seen = set()
    states = [s for s in states if not (_norm(s) in seen or seen.add(_norm(s)))]

    # 3. fetch + parse
    payload = fetch_hourly_state(session, base, years, states, args.demand, args.timeout)
    rows = parse_hourly_state(payload, args.state, args.demand, fetched_at)
    if not rows:
        raise SystemExit(f"ERROR: no '{args.state}' rows in response "
                         f"(keys seen: {_response_keys(payload)}).")

    # 4. persist
    added, updated, total = save_rows(args.csv, rows, args.append)
    print(f"{fetched_at}  parsed {len(rows)} rows  "
          f"(+{added} new, ~{updated} updated, {total} total) -> {args.csv}")
    return 0


def _response_keys(payload):
    keys = set()
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                keys.update(item.keys())
    return sorted(keys)


if __name__ == "__main__":
    sys.exit(main())
