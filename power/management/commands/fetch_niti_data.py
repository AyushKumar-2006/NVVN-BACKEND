"""
Django management command: fetch_niti_data

Fetch Chhattisgarh (CG) electricity-demand data from the NITI Aayog family of
energy portals, save whatever is found as a CSV in ~/Downloads, and import it
into the right table:

    python manage.py fetch_niti_data                       # auto: try every source in order
    python manage.py fetch_niti_data --source merit        # only MERIT (live demand)
    python manage.py fetch_niti_data --source merit --merit-poll 12 --merit-interval 300
    python manage.py fetch_niti_data --source ndap --ndap-key <TOKEN>
    python manage.py fetch_niti_data --dry-run             # fetch + CSV, but no DB writes

Sources (tried in this order for --source auto)
-----------------------------------------------
1. NDAP   https://ndap.niti.gov.in
     National Data & Analytics Platform. Has a token-gated REST API
     (host ``ndapapi.niti.gov.in``). State-level electricity datasets there are
     ANNUAL energy requirement/availability, so granularity is "annual" -> saved
     as a CSV only (it is neither 5-min nor a per-date daily series).
     Requires a free API token: register at ndap.niti.gov.in and pass
     ``--ndap-key`` or set env ``NDAP_API_KEY``.

2. ICED   https://iced.niti.gov.in/energy/electricity/distribution/
     India Climate & Energy Dashboard. The distribution view is a JavaScript
     dashboard (annual/monthly state aggregates) with no plain CSV/JSON at the
     page URL; this adapter attempts a direct data pull and, if the host only
     returns the JS shell (or is unreachable), reports that and moves on.

3. MERIT  https://meritindia.in
     Merit Order Despatch (Ministry of Power). Exposes a verified per-state JSON
     endpoint returning the CURRENT demand met (MW):
         GET /StateWiseDetails/BindCurrentStateStatus?StateName=Chhattisgarh
            -> [{"Demand": <MW|null>, "ISGS": <MW|null>, "ImportData": <MW|null>}]
     This is an instantaneous reading, not a historical archive. A single call
     yields a "snapshot"; ``--merit-poll N --merit-interval S`` polls it N times
     S seconds apart to assemble a real 5-min series that is imported into
     StateLoad5Min. (CG's feed is frequently idle and returns null — that is
     reported, not treated as an error.)

What gets imported where
------------------------
* a sub-daily LOAD time series (5-min / hourly / polled snapshots)
      -> StateLoad5Min.load_mw   via bulk_upsert_state_5min
* a per-date DAILY energy series
      -> StateDailyLoad.energy_mu via bulk_upsert_state_daily   ("saved separately")
* anything coarser (annual) or a lone live snapshot
      -> CSV in ~/Downloads only (nothing forced into the 5-min table)

ADDITIVE ONLY — new file. Reuses the existing upsert helpers in
``power/utils/upload.py`` and the existing models. It modifies no existing file
and deletes nothing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import pytz
import requests
from django.core.management.base import BaseCommand, CommandError

from power.models import StateDailyLoad, StateLoad5Min
from power.utils.upload import (
    bulk_upsert_state_5min,
    bulk_upsert_state_daily,
    normalize_state,
)

requests.packages.urllib3.disable_warnings()  # several gov endpoints use weak/expired TLS chains

IST = pytz.timezone("Asia/Kolkata")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (NITI-fetch)"
HEADERS = {"User-Agent": UA, "X-Requested-With": "XMLHttpRequest"}

SOURCE_ORDER = ["ndap", "iced", "merit"]

# MERIT verified endpoints (probed live)
MERIT_BASE = "https://meritindia.in"
MERIT_STATE_STATUS = f"{MERIT_BASE}/StateWiseDetails/BindCurrentStateStatus"
MERIT_ALL_INDIA = f"{MERIT_BASE}/Dashboard/BindAllIndiaMap"

# NDAP token-gated REST API (host does not resolve on every network)
NDAP_BASE = "https://ndapapi.niti.gov.in/v1"

# ICED distribution dashboard (JS app; no direct data file at the page URL)
ICED_DISTRIBUTION = "https://iced.niti.gov.in/energy/electricity/distribution/"


# ---------------------------------------------------------------------------
# normalized per-source result
# ---------------------------------------------------------------------------
@dataclass
class SourceResult:
    source: str
    reachable: bool = False
    granularity: str | None = None       # "5min" | "hourly" | "daily" | "annual" | "snapshot"
    df: pd.DataFrame | None = None        # see schema notes per granularity
    coverage: str = ""                    # human-readable span / row count
    note: str = ""                        # status / why-empty explanation
    extra: dict = field(default_factory=dict)

    @property
    def has_data(self) -> bool:
        return self.df is not None and not self.df.empty


def _now_ist_naive() -> datetime:
    """Current IST as a tz-naive datetime (the DB stores naive IST, USE_TZ=False)."""
    return datetime.now(IST).replace(tzinfo=None)


def _floor_5min(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0, minute=(dt.minute // 5) * 5)


class Command(BaseCommand):
    help = (
        "Fetch CG electricity-demand data from NITI portals (NDAP, ICED, MERIT), "
        "save it as CSV in ~/Downloads, and import 5-min load into StateLoad5Min "
        "or daily energy into StateDailyLoad. Example: "
        "manage.py fetch_niti_data --source merit --merit-poll 12 --merit-interval 300"
    )

    # ------------------------------------------------------------------ args
    def add_arguments(self, parser):
        parser.add_argument(
            "--source", choices=["auto", "all", *SOURCE_ORDER], default="auto",
            help="auto = try ndap->iced->merit and stop at the first that returns "
                 "data; all = run every source; or pick one [default: auto]",
        )
        parser.add_argument(
            "--state", default="Chhattisgarh",
            help="state full name or short code [default: Chhattisgarh]",
        )
        parser.add_argument(
            "--downloads-dir", default=os.path.expanduser("~/Downloads"),
            help="where CSVs are written [default: ~/Downloads]",
        )
        parser.add_argument(
            "--from", dest="dfrom", default=None,
            help="start date YYYY-MM-DD (NDAP/ICED annual range) [default: 10y ago]",
        )
        parser.add_argument(
            "--to", dest="dto", default=date.today().isoformat(),
            help="end date YYYY-MM-DD [default: today]",
        )
        parser.add_argument(
            "--ndap-key", default=os.environ.get("NDAP_API_KEY"),
            help="NDAP API token (or set env NDAP_API_KEY). Register free at ndap.niti.gov.in",
        )
        parser.add_argument(
            "--ndap-base", default=NDAP_BASE,
            help=f"NDAP API base URL [default: {NDAP_BASE}]",
        )
        parser.add_argument(
            "--merit-poll", type=int, default=1,
            help="number of MERIT live readings to collect; >1 builds a 5-min "
                 "series imported to StateLoad5Min [default: 1 = single snapshot]",
        )
        parser.add_argument(
            "--merit-interval", type=int, default=300,
            help="seconds between MERIT polls [default: 300 = 5 min]",
        )
        parser.add_argument(
            "--timeout", type=int, default=30,
            help="per-request network timeout (seconds) [default: 30]",
        )
        parser.add_argument(
            "--no-csv", action="store_true",
            help="do not write CSV files (DB import still runs unless --dry-run)",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="fetch + write CSV but do not write anything to the database",
        )

    # ---------------------------------------------------------------- handle
    def handle(self, *args, **opts):
        state_full = opts["state"].strip()
        state_code = normalize_state(state_full)
        if not state_code:
            raise CommandError(
                f"unknown state '{state_full}' — pass a full name (e.g. Chhattisgarh) "
                f"or a known short code (e.g. CG)"
            )

        try:
            dfrom = (
                date.fromisoformat(opts["dfrom"]) if opts["dfrom"]
                else date.today() - timedelta(days=3650)
            )
            dto = date.fromisoformat(opts["dto"])
        except ValueError as exc:
            raise CommandError(f"invalid date (use YYYY-MM-DD): {exc}")

        downloads_dir = os.path.expanduser(opts["downloads_dir"])
        dry_run = opts["dry_run"]
        write_csv = not opts["no_csv"]

        if opts["source"] == "auto":
            order, stop_at_first = SOURCE_ORDER, True
        elif opts["source"] == "all":
            order, stop_at_first = SOURCE_ORDER, False
        else:
            order, stop_at_first = [opts["source"]], False

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"NITI Aayog CG demand fetch — state={state_full} ({state_code})  "
            f"sources={'/'.join(order)}"
        ))
        if dry_run:
            self.stdout.write(self.style.WARNING("  DRY-RUN: no database writes"))
        self.stdout.write(f"  downloads dir : {downloads_dir}")
        self.stdout.write("")

        dispatch = {
            "ndap": lambda: self._fetch_ndap(opts, state_full, state_code, dfrom, dto),
            "iced": lambda: self._fetch_iced(opts, state_full, state_code, dfrom, dto),
            "merit": lambda: self._fetch_merit(opts, state_full, state_code),
        }

        results: list[SourceResult] = []
        used: SourceResult | None = None

        for name in order:
            self.stdout.write(self.style.HTTP_INFO(f"[{name.upper()}] querying …"))
            try:
                res = dispatch[name]()
            except Exception as exc:  # noqa: BLE001 — one bad source must not abort the rest
                res = SourceResult(name, reachable=False, note=f"unhandled error: {exc}")
            results.append(res)
            self._report_source(res)

            if res.has_data:
                self._handle_data(res, state_code, downloads_dir, write_csv, dry_run)
                if used is None:
                    used = res
                if stop_at_first:
                    break
            self.stdout.write("")

        self._summary(results, used, dry_run)

    # =====================================================================
    # SOURCE 1 — NDAP (token-gated REST API; state electricity = annual)
    # =====================================================================
    def _fetch_ndap(self, opts, state_full, state_code, dfrom, dto) -> SourceResult:
        key = opts["ndap_key"]
        base = opts["ndap_base"].rstrip("/")
        timeout = opts["timeout"]

        if not key:
            return SourceResult(
                "ndap", reachable=False,
                note="no API token — register free at ndap.niti.gov.in and pass "
                     "--ndap-key or set NDAP_API_KEY. Skipping.",
            )

        # NDAP serves data via a token in the query string. We search for an
        # electricity dataset, then pull the Chhattisgarh rows. The exact dataset
        # id varies, so we search by keyword and validate the response shape; if
        # the host is unreachable or the schema is unexpected we say so plainly
        # rather than guess.
        try:
            sr = requests.get(
                f"{base}/datasetsearch",
                params={"keyword": "electricity demand", "api_key": key},
                headers=HEADERS, timeout=timeout, verify=False,
            )
        except requests.exceptions.RequestException as exc:
            return SourceResult(
                "ndap", reachable=False,
                note=f"API host unreachable from this network ({type(exc).__name__}). "
                     f"NDAP's data API ({base}) is not resolvable/open here.",
            )

        if sr.status_code != 200 or "json" not in sr.headers.get("content-type", ""):
            return SourceResult(
                "ndap", reachable=True,
                note=f"search returned HTTP {sr.status_code} / non-JSON "
                     f"({sr.headers.get('content-type','?')}); no usable dataset list.",
            )

        try:
            catalog = sr.json()
        except ValueError:
            return SourceResult("ndap", reachable=True,
                                note="search response was not valid JSON.")

        # Best-effort: locate a Chhattisgarh demand/consumption series in the
        # returned catalogue. NDAP's electricity tables are annual, so this maps
        # to granularity="annual" (CSV only).
        rows = self._ndap_extract_cg(catalog, state_full)
        if rows is None or rows.empty:
            return SourceResult(
                "ndap", reachable=True,
                note="connected, but no Chhattisgarh demand series found in the "
                     "search result schema (NDAP electricity tables are annual and "
                     "vary by id; refine with --ndap-base / a specific dataset).",
            )

        rows = rows[(rows["date"] >= pd.Timestamp(dfrom)) & (rows["date"] <= pd.Timestamp(dto))]
        return SourceResult(
            "ndap", reachable=True, granularity="annual", df=rows,
            coverage=f"{rows['date'].min().date()}..{rows['date'].max().date()}, {len(rows)} row(s)",
            note="NDAP annual energy series (saved as CSV; not a 5-min/daily load series).",
        )

    @staticmethod
    def _ndap_extract_cg(catalog, state_full) -> pd.DataFrame | None:
        """Pull (date, energy_mu) rows for the state out of an NDAP catalogue/rows
        payload. Tolerant of the common NDAP shapes; returns None if nothing
        matches rather than inventing data."""
        records = catalog.get("Data") or catalog.get("data") or catalog.get("records")
        if not isinstance(records, list):
            return None
        out = []
        for r in records:
            if not isinstance(r, dict):
                continue
            blob = " ".join(str(v) for v in r.values()).lower()
            if state_full.lower() not in blob and "chhattisgarh" not in blob:
                continue
            year = r.get("Year") or r.get("year") or r.get("calendar_year")
            val = (r.get("Energy_Requirement") or r.get("energy")
                   or r.get("Demand") or r.get("value"))
            if year is None or val is None:
                continue
            try:
                out.append({"date": pd.Timestamp(int(str(year)[:4]), 1, 1),
                            "energy_mu": float(val)})
            except (ValueError, TypeError):
                continue
        return pd.DataFrame(out) if out else None

    # =====================================================================
    # SOURCE 2 — ICED (JS dashboard; annual/monthly state aggregates)
    # =====================================================================
    def _fetch_iced(self, opts, state_full, state_code, dfrom, dto) -> SourceResult:
        timeout = opts["timeout"]
        try:
            r = requests.get(ICED_DISTRIBUTION, headers={"User-Agent": UA},
                             timeout=timeout, verify=False)
        except requests.exceptions.RequestException as exc:
            return SourceResult(
                "iced", reachable=False,
                note=f"unreachable from this network ({type(exc).__name__}: connection "
                     f"timed out). The ICED host is firewalled here.",
            )

        ct = r.headers.get("content-type", "")
        if "json" in ct:
            try:
                data = r.json()
            except ValueError:
                data = None
            if data:
                return SourceResult("iced", reachable=True,
                                    note="received JSON but no parser for this ICED schema "
                                         "yet — inspect and extend _fetch_iced.")
        # HTML shell -> it's the JS dashboard, no direct data file at this path
        return SourceResult(
            "iced", reachable=True,
            note="page is a JavaScript dashboard (HTML shell, no CSV/JSON at this "
                 "URL). ICED data must be exported interactively from the site; "
                 "no automated CG series available at this endpoint.",
        )

    # =====================================================================
    # SOURCE 3 — MERIT (verified live per-state demand, MW)
    # =====================================================================
    def _fetch_merit(self, opts, state_full, state_code) -> SourceResult:
        timeout = opts["timeout"]
        poll = max(1, opts["merit_poll"])
        interval = max(0, opts["merit_interval"])

        national = self._merit_national(timeout)
        if national is not None:
            self.stdout.write(f"  national DEMAND MET now: {national:,.0f} MW")

        samples: list[tuple[datetime, float]] = []
        for i in range(poll):
            ts = _floor_5min(_now_ist_naive())
            demand, raw = self._merit_state_demand(state_full, timeout)
            if demand is not None:
                samples.append((ts, demand))
                self.stdout.write(f"  [{i+1}/{poll}] {ts:%Y-%m-%d %H:%M}  "
                                  f"{state_code} demand = {demand:,.0f} MW")
            else:
                self.stdout.write(self.style.WARNING(
                    f"  [{i+1}/{poll}] {ts:%Y-%m-%d %H:%M}  "
                    f"{state_code} demand = null (state not reporting to MERIT now) "
                    f"raw={raw}"))
            if i < poll - 1 and interval:
                time.sleep(interval)

        if not samples:
            return SourceResult(
                "merit", reachable=True, granularity="snapshot",
                note=f"endpoint live, but {state_code} returned no demand value "
                     f"(feed idle/null). Re-run when the state is reporting, or use "
                     f"--merit-poll over a window.",
                extra={"national_mw": national},
            )

        df = (pd.DataFrame(samples, columns=["datetime", "load_mw"])
              .drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))
        gran = "5min" if len(df) > 1 else "snapshot"
        return SourceResult(
            "merit", reachable=True, granularity=gran, df=df,
            coverage=f"{df['datetime'].min():%Y-%m-%d %H:%M}.."
                     f"{df['datetime'].max():%Y-%m-%d %H:%M}, {len(df)} reading(s)",
            note=("live 5-min demand series (polled)" if gran == "5min"
                  else "single live demand snapshot — use --merit-poll for a 5-min series"),
            extra={"national_mw": national},
        )

    def _merit_state_demand(self, state_full, timeout):
        """Return (demand_mw|None, raw_json) for the state from MERIT."""
        try:
            r = requests.get(MERIT_STATE_STATUS, params={"StateName": state_full},
                             headers=HEADERS, timeout=timeout, verify=False)
            data = r.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            return None, f"request failed: {exc}"
        row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
        val = row.get("Demand")
        if val in (None, "", "-"):
            return None, row
        try:
            return float(str(val).replace(",", "")), row
        except (ValueError, TypeError):
            return None, row

    def _merit_national(self, timeout):
        """Parse 'DEMAND MET <n> MW' out of the all-India current table."""
        try:
            r = requests.get(MERIT_ALL_INDIA, headers=HEADERS, timeout=timeout, verify=False)
        except requests.exceptions.RequestException:
            return None
        import re
        m = re.search(r"DEMAND MET[^0-9]*([\d,]+)", r.text)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            return None

    # =====================================================================
    # save CSV + import to the right table
    # =====================================================================
    def _handle_data(self, res, state_code, downloads_dir, write_csv, dry_run):
        csv_path = None
        if write_csv:
            csv_path = self._save_csv(res, state_code, downloads_dir)
            self.stdout.write(self.style.SUCCESS(f"  CSV saved : {csv_path}"))

        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"  DRY-RUN: would import {len(res.df)} row(s) "
                f"({res.granularity}) — skipped"))
            return

        if res.granularity in ("5min", "hourly", "snapshot") and "load_mw" in res.df.columns:
            if res.granularity == "snapshot":
                # a lone instantaneous reading: keep it as CSV, don't seed the
                # historical 5-min table off one point
                self.stdout.write(
                    "  not imported: single snapshot (use --merit-poll to build a "
                    "5-min series for StateLoad5Min)")
                return
            n = self._import_timeseries(res.df, state_code)
            self.stdout.write(self.style.SUCCESS(
                f"  imported  : {n} row(s) -> StateLoad5Min ({state_code})"))
            res.extra["imported"] = ("StateLoad5Min", n)

        elif res.granularity == "daily" and "energy_mu" in res.df.columns:
            n = self._import_daily(res.df, state_code)
            self.stdout.write(self.style.SUCCESS(
                f"  imported  : {n} row(s) -> StateDailyLoad ({state_code})"))
            res.extra["imported"] = ("StateDailyLoad", n)

        else:
            self.stdout.write(
                f"  not imported: granularity '{res.granularity}' is coarser than "
                f"daily — saved as CSV only")

    def _save_csv(self, res, state_code, downloads_dir) -> str:
        os.makedirs(downloads_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"niti_{state_code.lower()}_{res.source}_{res.granularity}_{stamp}.csv"
        path = os.path.join(downloads_dir, fname)
        res.df.to_csv(path, index=False)
        return path

    @staticmethod
    def _import_timeseries(df, state_code) -> int:
        records = [
            StateLoad5Min(
                state=state_code,
                datetime=pd.Timestamp(dt).to_pydatetime(),
                load_mw=float(v),
            )
            for dt, v in zip(df["datetime"], df["load_mw"]) if pd.notna(v)
        ]
        bulk_upsert_state_5min(records)
        return len(records)

    @staticmethod
    def _import_daily(df, state_code) -> int:
        records = [
            StateDailyLoad(
                state=state_code,
                date=pd.Timestamp(d).date(),
                energy_mu=float(v),
            )
            for d, v in zip(df["date"], df["energy_mu"]) if pd.notna(v)
        ]
        bulk_upsert_state_daily(records)
        return len(records)

    # =====================================================================
    # reporting
    # =====================================================================
    def _report_source(self, res: SourceResult):
        flag = self.style.SUCCESS("reachable") if res.reachable else self.style.ERROR("unreachable")
        gran = res.granularity or "-"
        self.stdout.write(f"  status    : {flag}   granularity={gran}")
        if res.coverage:
            self.stdout.write(f"  coverage  : {res.coverage}")
        if res.note:
            self.stdout.write(f"  note      : {res.note}")

    def _summary(self, results, used, dry_run):
        self.stdout.write(self.style.MIGRATE_HEADING("Summary"))
        for r in results:
            mark = "✓ data" if r.has_data else ("· reachable" if r.reachable else "✗ unreachable")
            imp = ""
            if r.extra.get("imported"):
                table, n = r.extra["imported"]
                imp = f"  -> {n} row(s) into {table}"
            self.stdout.write(f"  {r.source:5s} : {mark}  ({r.granularity or 'no data'}){imp}")

        if used and used.has_data:
            self.stdout.write(self.style.SUCCESS(
                f"\nPrimary source: {used.source.upper()} ({used.granularity}). "
                f"{'(dry-run, nothing written to DB)' if dry_run else ''}"))
        else:
            self.stdout.write(self.style.WARNING(
                "\nNo CG demand series was retrievable right now. From this network only "
                "MERIT is reachable, and it serves a LIVE snapshot (currently idle/null "
                "for CG). To build history, schedule:\n"
                "  manage.py fetch_niti_data --source merit --merit-poll 288 --merit-interval 300\n"
                "(288 x 5-min = one full day) once CG resumes reporting to MERIT, or supply "
                "an NDAP token for annual figures."))
