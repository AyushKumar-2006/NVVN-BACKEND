"""
WRLDC daily PSP report: crawler + downloader + Chhattisgarh demand parser.

Source (verified live, June 2026)
---------------------------------
    https://reporting.wrldc.in:8081/PSPExcel/{YEAR}/{MonthName}/WRLDC_PSP_Report_{DD-MM-YYYY}.xls

Facts established by probing the live server (so this keeps working / is honest):
  * Served over **HTTPS on port 8081**. Plain :80 is closed and the often-quoted
    ``http://reporting.wrldc.in/PSP/`` path 404s — the real path is
    ``:8081/PSPExcel/``.
  * Files are legacy OLE2 ``.xls`` (BIFF). The year and month folders are
    directory-listable (IIS autoindex), so we enumerate exact filenames instead
    of guessing dates.
  * Years available: **2019 .. current** (there is no 2018 folder).
  * The server frequently delivers **TRUNCATED bodies** (bytes received <
    Content-Length), which makes a perfectly-good ``.xls`` look corrupt
    ("Can't read SAT" / "Expected BOF record"). Downloads are therefore
    size-verified *and* re-opened to validate, and truncated deliveries are
    retried.

What CG data the report actually contains
-----------------------------------------
The PSP report is a DAILY operations report. The only REAL Chhattisgarh demand
(MW) points it carries live in workbook sections 2(B) and 2(C):
  * Off-Peak (03:00) Demand Met
  * Evening Peak (19:00) Demand Met
  * Maximum Demand Met of the day  (+ the clock time it occurred)
=> a sparse (~3 points/day) but REAL intraday demand series. There is **no**
15-min / 5-min block-wise load curve anywhere in the report.

ADDITIVE ONLY — new file. Pure ``requests`` + ``xlrd`` (no Django imports), so
the parser can be unit-tested standalone.
"""
from __future__ import annotations

import calendar
import os
import re
from datetime import date, datetime, time as dtime

import requests
import xlrd

requests.packages.urllib3.disable_warnings()  # gov TLS cert -> verify=False

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
ROOT = "https://reporting.wrldc.in:8081"
BASE = ROOT + "/PSPExcel"
HEADERS = {"User-Agent": "Mozilla/5.0 (WRLDC PSP ingestion; +local)"}

STATE_CODE = "CG_WRLDC"                       # kept separate from synthetic 'CG'
STATE_NAMES = ("chhattisgarh", "chhatisgarh", "chattisgarh")

# Plausible CG state demand (MW) — used to pick the correct numeric token out of
# a row that also holds shortage / requirement / energy-MU figures.
MW_RANGE = (300.0, 15000.0)

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0"
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?$")
_HREF_RE = re.compile(r'href="([^"]+)"', re.I)
_FNAME_DATE_RE = re.compile(r"WRLDC_PSP_Report_(\d{2})-(\d{2})-(\d{4})\.xls$", re.I)


class _Null:
    """Swallow xlrd's noisy warning output."""
    def write(self, *_a, **_k):
        pass
    def flush(self):
        pass


_NULL = _Null()


# --------------------------------------------------------------------------
# URL construction + directory crawl
# --------------------------------------------------------------------------
def month_name(m: int) -> str:
    return calendar.month_name[m]            # 1 -> "January"


def psp_url(d: date) -> str:
    fname = f"WRLDC_PSP_Report_{d.day:02d}-{d.month:02d}-{d.year}.xls"
    return f"{BASE}/{d.year}/{month_name(d.month)}/{fname}"


def _listing(url: str, session: requests.Session) -> list[str]:
    try:
        r = session.get(url, headers=HEADERS, timeout=40, verify=False)
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return _HREF_RE.findall(r.text)


def list_years(session: requests.Session | None = None) -> list[int]:
    session = session or requests.Session()
    out = []
    for h in _listing(BASE + "/", session):
        m = re.search(r"/PSPExcel/(\d{4})/?$", h)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def list_year_files(year: int, session: requests.Session | None = None) -> list[tuple[date, str]]:
    """Crawl ``/PSPExcel/{year}/`` and every month folder under it.

    Returns a date-sorted, de-duplicated list of ``(report_date, file_url)``.
    """
    session = session or requests.Session()
    found: dict[date, str] = {}
    for h in _listing(f"{BASE}/{year}/", session):
        mm = re.search(rf"/PSPExcel/{year}/([^/\"]+)/?$", h)
        if not mm or "." in mm.group(1):     # skip "[To Parent Directory]" etc.
            continue
        month = mm.group(1)
        for fh in _listing(f"{BASE}/{year}/{month}/", session):
            fm = _FNAME_DATE_RE.search(fh)
            if not fm:
                continue
            dd, mo, yy = int(fm.group(1)), int(fm.group(2)), int(fm.group(3))
            try:
                d = date(yy, mo, dd)
            except ValueError:
                continue
            found[d] = f"{ROOT}{fh}" if fh.startswith("/") else \
                f"{BASE}/{year}/{month}/{os.path.basename(fh)}"
    return sorted(found.items())


# --------------------------------------------------------------------------
# Download (size-verified + open-verified, because the server truncates bodies)
# --------------------------------------------------------------------------
def is_good_xls(path: str) -> bool:
    """True iff ``path`` is a complete, openable OLE2 .xls workbook."""
    try:
        if os.path.getsize(path) < 2000:
            return False
        with open(path, "rb") as fh:
            if fh.read(4) != _OLE2_MAGIC:
                return False
        wb = xlrd.open_workbook(path, logfile=_NULL, on_demand=True)
        return wb.nsheets > 0
    except Exception:                        # noqa: BLE001
        return False


def download(url: str, dest: str, session: requests.Session | None = None,
             retries: int = 4) -> str:
    """Download one PSP .xls to ``dest``, retrying truncated deliveries.

    Returns one of: 'cached' | 'downloaded' | 'missing' | 'corrupt' | 'error'.
    A pre-existing, valid file is kept (never re-downloaded, never deleted).
    """
    session = session or requests.Session()
    if os.path.exists(dest) and is_good_xls(dest):
        return "cached"

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = "error"
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=120, verify=False)
        except requests.RequestException:
            last = "error"
            continue
        if r.status_code == 404:
            return "missing"
        if r.status_code != 200:
            last = "error"
            continue

        body = r.content
        if body[:4] != _OLE2_MAGIC:
            # a 404 HTML page or other non-xls payload
            head = body[:400].lstrip().lower()
            return "missing" if head[:1] == b"<" or b"not found" in head else "corrupt"

        clen = r.headers.get("Content-Length")
        if clen and len(body) < int(clen):
            last = "corrupt"                 # truncated body -> retry
            continue

        tmp = dest + ".part"
        with open(tmp, "wb") as fh:
            fh.write(body)
        if is_good_xls(tmp):
            os.replace(tmp, dest)
            return "downloaded"
        try:
            os.remove(tmp)
        except OSError:
            pass
        last = "corrupt"
    return last


# --------------------------------------------------------------------------
# Parse: Chhattisgarh demand points out of one workbook
# --------------------------------------------------------------------------
def _cell(sh, r, c):
    try:
        return sh.cell_value(r, c)
    except IndexError:
        return ""


def _s(v) -> str:
    return str(v).strip()


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _parse_time(v):
    """``datetime.time`` from 'HH:MM' / 'HH:MM:SS' text or an Excel time float."""
    if isinstance(v, float) and 0.0 < v < 1.0:
        secs = round(v * 86400)
        return dtime((secs // 3600) % 24, (secs % 3600) // 60)
    m = _TIME_RE.match(_s(v))
    if m:
        return dtime(int(m.group(1)), int(m.group(2)))
    return None


def _snap5(t: dtime) -> dtime:
    """Snap a clock time to the nearest 5-minute grid point."""
    total = (t.hour * 60 + t.minute + 2) // 5 * 5
    total %= 1440
    return dtime(total // 60, total % 60)


def _first_numeric(sh, r, lo_col, hi_col, rng=MW_RANGE):
    lo, hi = rng
    for c in range(lo_col, min(hi_col, sh.ncols)):
        v = _num(_cell(sh, r, c))
        if v is not None and lo <= v <= hi:
            return v
    return None


def _find_sections(sh) -> dict[str, int]:
    sec: dict[str, int] = {}
    for r in range(min(sh.nrows, 90)):
        t = _s(_cell(sh, r, 0)).lower()
        if not t:
            continue
        if "2(b)" in t or ("demand met in mw" in t and "forecast" in t):
            sec.setdefault("2B", r)
        if "2(c)" in t or "maximum demand met" in t:
            sec.setdefault("2C", r)
        if "3(a)" in t or "entities generation" in t:
            sec.setdefault("3A", r)
    return sec


def _cg_row(sh, lo: int, hi: int):
    for r in range(max(lo, 0), min(hi, sh.nrows)):
        if any(n in _s(_cell(sh, r, 0)).lower() for n in STATE_NAMES):
            return r
    return None


def _peak_offpeak_cols(sh, lo: int):
    """Column of the 'Evening Peak' and 'Off-Peak' headers within the 2(B) block."""
    ev = op = None
    for r in range(lo, min(lo + 6, sh.nrows)):
        for c in range(sh.ncols):
            t = _s(_cell(sh, r, c)).lower()
            if not t:
                continue
            if ev is None and "evening peak" in t:
                ev = c
            if op is None and ("off-peak" in t or "off peak" in t):
                op = c
    # fall back to the literal clock-time headers if the labels moved
    if ev is None or op is None:
        for r in range(lo, min(lo + 6, sh.nrows)):
            for c in range(sh.ncols):
                t = _s(_cell(sh, r, c))
                if ev is None and t == "19:00":
                    ev = c
                if op is None and t == "03:00":
                    op = c
    return ev, op


def extract_cg_points(xls_path: str, report_date: date) -> list[dict]:
    """Extract CG demand (MW) points from one PSP workbook.

    Returns ``[{datetime, load_mw, kind}, ...]`` (deduped by timestamp, larger
    value wins). ``kind`` is one of 'max' | 'evening_peak' | 'off_peak'.
    """
    try:
        wb = xlrd.open_workbook(xls_path, logfile=_NULL)
        sh = wb.sheet_by_index(0)
    except Exception:                        # noqa: BLE001 — bad/corrupt file
        return []

    sec = _find_sections(sh)
    points: dict[datetime, tuple[float, str]] = {}

    def put(dt, mw, kind):
        if mw is None:
            return
        if dt not in points or mw > points[dt][0]:
            points[dt] = (float(mw), kind)

    # ---- 2(C): maximum demand met of the day (+ time) — the headline figure
    if "2C" in sec:
        end = sec.get("3A", sh.nrows)
        r = _cg_row(sh, sec["2C"] + 1, end)
        if r is not None:
            mw = _first_numeric(sh, r, 1, sh.ncols)
            t = None
            for c in range(1, sh.ncols):
                t = _parse_time(_cell(sh, r, c))
                if t is not None:
                    break
            if mw is not None and t is not None:
                put(datetime.combine(report_date, _snap5(t)), mw, "max")

    # ---- 2(B): off-peak (03:00) + evening-peak (19:00) demand met
    if "2B" in sec:
        end = sec.get("2C", sh.nrows)
        r = _cg_row(sh, sec["2B"] + 1, end)
        ev_col, op_col = _peak_offpeak_cols(sh, sec["2B"])
        if r is not None and ev_col is not None and op_col is not None and ev_col < op_col:
            put(datetime.combine(report_date, dtime(19, 0)),
                _first_numeric(sh, r, ev_col, op_col), "evening_peak")
            put(datetime.combine(report_date, dtime(3, 0)),
                _first_numeric(sh, r, op_col, sh.ncols), "off_peak")

    return [{"datetime": dt, "load_mw": round(v, 2), "kind": k}
            for dt, (v, k) in sorted(points.items())]
