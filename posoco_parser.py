#!/usr/bin/env python
"""
posoco_parser.py  —  Ingest real DAILY load data from POSOCO / Grid-India
                     "Power Supply Position" (PSP) daily PDF reports.

What it does
------------
- Reads a daily PSP report PDF (local file or downloaded URL).
- Extracts Chhattisgarh's daily *peak demand* (MW) and *energy met* (MU).
- Saves the energy figure to the existing `StateDailyLoad` table (the project's
  existing daily format), reusing `power.utils.upload.bulk_upsert_state_daily`.
- Also writes a CSV (date, peak_mw, energy_mu) so the peak is not lost.

Why
---
Gives a real ~10-year DAILY series for the Prophet model instead of the
synthetic 2023-2025 data.

IMPORTANT — this only ADDS a new file. It does not modify any existing code.
It reuses existing models / helpers read-only.

Honest caveats (please read)
----------------------------
1. The exact daily PSP **download URL pattern** on grid-india.in / posoco.in
   changes over time and the archive page is dynamic. The downloader below uses
   configurable URL templates (`REPORT_URL_TEMPLATES`) — verify/adjust them, or
   just run in `--pdf <file>` mode on a report you downloaded by hand.
2. The PSP **table layout** (column order) has changed across report eras. The
   parser extracts the "Chhattisgarh" row and reads numeric columns by index
   (`PEAK_COL`, `ENERGY_COL`); run with `-v` to print the raw parsed row and
   calibrate these two constants for your report vintage.
3. `StateDailyLoad` has only an `energy_mu` column (no peak field). To respect
   "do not modify existing files", peak is exported to CSV rather than stored in
   the DB. The current Prophet trainer reads `StateLoad5Min`; pointing it at
   `StateDailyLoad` is a separate, future change (not done here).

Usage
-----
    # parse one already-downloaded PDF
    python posoco_parser.py --pdf ~/Downloads/PSP_2023-05-15.pdf --date 2023-05-15

    # try to download + parse a date range (verify REPORT_URL_TEMPLATES first)
    python posoco_parser.py --from 2023-05-01 --to 2023-05-31

    # parse every *.pdf in a folder (date taken from each filename if possible)
    python posoco_parser.py --dir ~/Downloads/psp_reports/

    # see what would be saved without writing to the DB
    python posoco_parser.py --pdf report.pdf --date 2023-05-15 --dry-run -v
"""

import argparse
import csv
import os
import re
import sys
from datetime import date, datetime, timedelta

import requests

# --------------------------------------------------------------------------
# Django bootstrap (script lives at the project root)
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django  # noqa: E402

django.setup()

import fitz  # PyMuPDF  # noqa: E402

from power.models import StateDailyLoad  # noqa: E402
from power.utils.upload import bulk_upsert_state_daily  # noqa: E402

# --------------------------------------------------------------------------
# Config — adjust to match the report vintage / state you want
# --------------------------------------------------------------------------
STATE_CODE = "CG"                 # internal short code stored in the DB
STATE_NAMES = ("Chhattisgarh", "Chhatisgarh", "Chattisgarh")  # PDF spellings

# Column index (0-based) of the numeric tokens on the Chhattisgarh row.
# Run with -v to print the row and set these for your report layout.
PEAK_COL = 5       # "Max Demand Met During the Day (MW)"
ENERGY_COL = 2     # "Energy Met (MU)"

# Plausible ranges used as a sanity heuristic / fallback (CG-scale).
PEAK_RANGE = (1000.0, 9000.0)     # MW
ENERGY_RANGE = (20.0, 300.0)      # MU/day

# Candidate download URL templates ({y}={year} {m}=month {d}=day {date}=YYYY-MM-DD)
# These WILL need verifying against the live site — prefer --pdf/--dir mode.
REPORT_URL_TEMPLATES = [
    "https://posoco.in/wp-content/uploads/{y}/{m}/{d}.{mon}.{y}_NLDC_PSP.pdf",
    "https://grid-india.in/wp-content/uploads/{y}/{m}/{d}.{mon}.{y}_NLDC_PSP.pdf",
]

OUT_CSV = "posoco_cg_daily.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (ingestion script; +local)"}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_pdf(report_date: date, dest_dir: str = "/tmp") -> str | None:
    """Best-effort download of a day's PSP PDF. Returns local path or None."""
    for tmpl in REPORT_URL_TEMPLATES:
        url = tmpl.format(
            y=report_date.year,
            m=f"{report_date.month:02d}",
            d=f"{report_date.day:02d}",
            mon=report_date.strftime("%b"),
            date=report_date.isoformat(),
        )
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, verify=False)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                path = os.path.join(dest_dir, f"PSP_{report_date.isoformat()}.pdf")
                with open(path, "wb") as fh:
                    fh.write(r.content)
                print(f"  downloaded {url}")
                return path
        except Exception as e:  # noqa: BLE001
            print(f"  (download failed: {url} -> {e})")
    print(f"  !! could not download PSP for {report_date} — "
          f"verify REPORT_URL_TEMPLATES or use --pdf/--dir")
    return None


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _nums(line: str):
    out = []
    for tok in _NUM.findall(line):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def _pick(nums, col, lo, hi):
    """Use the configured column if it is in range, else the first in-range value."""
    if 0 <= col < len(nums) and lo <= nums[col] <= hi:
        return nums[col]
    for v in nums:
        if lo <= v <= hi:
            return v
    return None


def extract_cg_daily(pdf_path: str, report_date: date, verbose: bool = False) -> dict | None:
    """Pull Chhattisgarh peak (MW) + energy (MU) from a PSP PDF."""
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    for raw in text.splitlines():
        if any(name.lower() in raw.lower() for name in STATE_NAMES):
            nums = _nums(raw)
            if verbose:
                print(f"  [Chhattisgarh row] {raw.strip()}")
                print(f"  [parsed numbers ] {nums}")
            if not nums:
                continue
            peak = _pick(nums, PEAK_COL, *PEAK_RANGE)
            energy = _pick(nums, ENERGY_COL, *ENERGY_RANGE)
            if peak is None and energy is None:
                continue
            return {"date": report_date, "peak_mw": peak, "energy_mu": energy}

    print(f"  !! Chhattisgarh row not found in {os.path.basename(pdf_path)}")
    return None


# --------------------------------------------------------------------------
# Save (reuses the existing daily upsert helper)
# --------------------------------------------------------------------------
def save_records(records: list[dict], dry_run: bool = False) -> int:
    """records: [{date, peak_mw, energy_mu}, ...] -> StateDailyLoad + CSV."""
    rows = [r for r in records if r and r.get("energy_mu") is not None]

    # CSV export keeps the peak (which StateDailyLoad has no column for)
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["date", "state", "peak_mw", "energy_mu"])
        for r in records:
            if r:
                w.writerow([r["date"].isoformat(), STATE_CODE,
                            r.get("peak_mw"), r.get("energy_mu")])

    if dry_run:
        print(f"  [dry-run] would upsert {len(rows)} StateDailyLoad rows")
        return 0

    db_records = [
        StateDailyLoad(state=STATE_CODE, date=r["date"], energy_mu=float(r["energy_mu"]))
        for r in rows
    ]
    if db_records:
        bulk_upsert_state_daily(db_records)   # existing pipeline helper
    return len(db_records)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _date_from_name(path: str):
    m = re.search(r"(\d{4})[-_.](\d{2})[-_.](\d{2})", os.path.basename(path))
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{2})[-_.]([A-Za-z]{3})[-_.](\d{4})", os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(0).replace("_", "-").replace(".", "-"),
                                     "%d-%b-%Y").date()
        except ValueError:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Parse POSOCO PSP PDFs -> StateDailyLoad (CG)")
    ap.add_argument("--pdf", help="path to a single PSP PDF")
    ap.add_argument("--dir", help="folder of PSP PDFs (date inferred from filename)")
    ap.add_argument("--date", help="report date YYYY-MM-DD (for --pdf)")
    ap.add_argument("--from", dest="dfrom", help="download+parse range start YYYY-MM-DD")
    ap.add_argument("--to", dest="dto", help="download+parse range end YYYY-MM-DD")
    ap.add_argument("--dry-run", action="store_true", help="parse but do not write to DB")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    records = []

    if args.pdf:
        d = date.fromisoformat(args.date) if args.date else _date_from_name(args.pdf)
        if not d:
            ap.error("could not determine report date — pass --date YYYY-MM-DD")
        print(f"Parsing {args.pdf} ({d})")
        records.append(extract_cg_daily(args.pdf, d, args.verbose))

    elif args.dir:
        for fn in sorted(os.listdir(args.dir)):
            if fn.lower().endswith(".pdf"):
                path = os.path.join(args.dir, fn)
                d = _date_from_name(fn)
                if not d:
                    print(f"  skip {fn}: no date in filename")
                    continue
                print(f"Parsing {fn} ({d})")
                records.append(extract_cg_daily(path, d, args.verbose))

    elif args.dfrom and args.dto:
        cur, end = date.fromisoformat(args.dfrom), date.fromisoformat(args.dto)
        while cur <= end:
            print(f"Fetching PSP for {cur}")
            path = download_pdf(cur)
            if path:
                records.append(extract_cg_daily(path, cur, args.verbose))
            cur += timedelta(days=1)
    else:
        ap.error("provide one of: --pdf, --dir, or --from/--to")

    parsed = [r for r in records if r]
    print(f"\nParsed {len(parsed)} day(s).")
    for r in parsed:
        print(f"  {r['date']}  peak={r['peak_mw']} MW  energy={r['energy_mu']} MU")

    saved = save_records(records, dry_run=args.dry_run)
    print(f"\nSaved {saved} row(s) to StateDailyLoad (CG). CSV -> {OUT_CSV}")


if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()  # quiet verify=False notices
    main()
