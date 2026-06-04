"""
Working downloader + parser for Grid-India (NLDC) daily PSP reports.

This is ADDITIVE — it does not modify the existing `posoco_parser.py`. It
provides the *verified* download URL and a real-PDF table parser that the
`fetch_posoco_data` management command uses.

Verified URL pattern (probed live, works 2013 -> present):
    https://report.grid-india.in/ReportData/Daily%20Report/PSP%20Report/
        {FY}/{Month}%20{YYYY}/{DD.MM.YY}_NLDC_PSP.pdf
    FY = Indian financial year folder (April-March), e.g. "2024-2025"

The PSP "state-wise" table row for Chhattisgarh looks like:
    [Chhattisgarh, 5364, 0, 110.6, 63.0, 1.8, 307, 0.00]
mapping to header:
    Max.Demand Met (MW)=5364  ...  Energy Met (MU)=110.6  ...
so peak = first numeric column, energy = third numeric column. We also
range-validate as a fallback for older report layouts.
"""

import calendar
import os
import re
from datetime import date

import requests

import fitz  # PyMuPDF

from power.models import StateDailyLoad
from power.utils.upload import bulk_upsert_state_daily

requests.packages.urllib3.disable_warnings()  # gov cert -> verify=False

STATE_CODE = "CG"
STATE_NAMES = ("chhattisgarh", "chhatisgarh", "chattisgarh")

BASE_URL = "https://report.grid-india.in/ReportData/Daily%20Report/PSP%20Report"
HEADERS = {"User-Agent": "Mozilla/5.0 (ingestion; +local)"}

# Sanity ranges for CG (Max Demand MW, Energy Met MU)
PEAK_RANGE = (800.0, 12000.0)
ENERGY_RANGE = (10.0, 400.0)
PEAK_COL, ENERGY_COL = 0, 2   # numeric-column indices in the modern layout


# --------------------------------------------------------------------------
# URL + download
# --------------------------------------------------------------------------
def financial_year(d: date) -> str:
    """Indian FY folder name (April-March)."""
    return f"{d.year}-{d.year + 1}" if d.month >= 4 else f"{d.year - 1}-{d.year}"


def psp_url(d: date) -> str:
    month = calendar.month_name[d.month]                 # "January"
    fname = f"{d.day:02d}.{d.month:02d}.{d.strftime('%y')}_NLDC_PSP.pdf"
    return f"{BASE_URL}/{financial_year(d)}/{month}%20{d.year}/{fname}"


def download_psp(d: date, dest_dir: str = "/tmp/psp_cache") -> str | None:
    """Download one day's PSP PDF (cached). Returns local path or None."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"PSP_{d.isoformat()}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    try:
        r = requests.get(psp_url(d), headers=HEADERS, timeout=40, verify=False)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            with open(path, "wb") as fh:
                fh.write(r.content)
            return path
    except Exception:  # noqa: BLE001
        return None
    return None


# --------------------------------------------------------------------------
# Parse (coordinate-based row extraction)
# --------------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?$")


def _row_numbers(page, names, tol: float = 4.0):
    """Numbers on the same horizontal band as the state name on a page."""
    words = page.get_text("words")  # (x0, y0, x1, y1, text, ...)
    for w in words:
        if any(n in w[4].lower() for n in names):
            yc = (w[1] + w[3]) / 2
            row = sorted(
                (x for x in words if abs((x[1] + x[3]) / 2 - yc) < tol),
                key=lambda x: x[0],
            )
            nums = [
                float(x[4].replace(",", ""))
                for x in row
                if _NUM.match(x[4].replace(",", ""))
            ]
            if nums:
                return nums
    return None


def _pick(nums, idx, lo, hi):
    if idx < len(nums) and lo <= nums[idx] <= hi:
        return nums[idx]
    for v in nums:
        if lo <= v <= hi:
            return v
    return None


def extract_cg(pdf_path: str) -> dict | None:
    """Extract Chhattisgarh {peak_mw, energy_mu} from a PSP PDF."""
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            nums = _row_numbers(page, STATE_NAMES)
            if nums:
                peak = _pick(nums, PEAK_COL, *PEAK_RANGE)
                energy = _pick(nums, ENERGY_COL, *ENERGY_RANGE)
                if peak is not None or energy is not None:
                    return {"peak_mw": peak, "energy_mu": energy, "raw": nums}
    finally:
        doc.close()
    return None


# --------------------------------------------------------------------------
# Save (reuses the existing daily upsert helper)
# --------------------------------------------------------------------------
def save_daily(records: list[dict], dry_run: bool = False) -> int:
    """records: [{date, peak_mw, energy_mu}] -> StateDailyLoad (energy_mu)."""
    rows = [r for r in records if r and r.get("energy_mu") is not None]
    if dry_run or not rows:
        return 0
    db_records = [
        StateDailyLoad(state=STATE_CODE, date=r["date"], energy_mu=float(r["energy_mu"]))
        for r in rows
    ]
    bulk_upsert_state_daily(db_records)   # existing pipeline helper
    return len(db_records)
