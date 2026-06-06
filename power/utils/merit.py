"""
Shared MERIT India fetch helpers for live state power demand.

MERIT (Merit Order Despatch of Electricity, Ministry of Power) publishes the
*current* demand met per state as JSON. Verified working endpoint (the one the
public dashboard actually calls):

    GET https://meritindia.in/StateWiseDetails/BindCurrentStateStatus?StateName=Chhattisgarh
        -> [{"Demand": <MW|null>, "ISGS": <MW|null>, "ImportData": <MW|null>}]

NOTE: the commonly-quoted ``/Dashboard/BindCurrentStateStatus`` path returns an
HTML error page — the working controller is ``/StateWiseDetails/...``. We try the
Dashboard path first (in case it is ever fixed) and fall back to the working one,
so callers always reach a live endpoint.

The feed is a LIVE snapshot, not an archive: ``Demand`` is frequently ``null``
when no state is reporting at that instant. ``null`` is a normal, expected value
(handled gracefully), not an error.

Import-safe without Django: ``fetch_state_demand`` is pure (``requests`` only).
``save_reading`` lazily imports the Django model, so the same fetch code is
reused by the standalone GitHub Actions collector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pytz
import requests

try:  # gov TLS chains are often weak/expired -> verify=False
    requests.packages.urllib3.disable_warnings()
except Exception:  # noqa: BLE001
    pass

IST = pytz.timezone("Asia/Kolkata")

DEFAULT_STATE_NAME = "Chhattisgarh"
DEFAULT_STATE_CODE = "CG"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) MERIT-poll",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://meritindia.in/",
}

# tried in order; first valid-JSON response wins
STATE_STATUS_ENDPOINTS = [
    "https://meritindia.in/Dashboard/BindCurrentStateStatus",        # as provided (currently errors)
    "https://meritindia.in/StateWiseDetails/BindCurrentStateStatus",  # verified JSON endpoint
]
ALL_INDIA_URL = "https://meritindia.in/Dashboard/BindAllIndiaMap"


@dataclass
class MeritReading:
    state_code: str
    state_name: str
    timestamp: datetime          # tz-naive IST, floored to the 5-min slot
    demand_mw: float | None
    isgs_mw: float | None = None
    import_mw: float | None = None
    endpoint: str | None = None
    ok: bool = False             # endpoint reachable & JSON parsed
    error: str | None = None

    @property
    def has_demand(self) -> bool:
        return self.demand_mw is not None


def now_ist() -> datetime:
    """Current IST as a tz-naive datetime (the DB stores naive IST; USE_TZ=False)."""
    return datetime.now(IST).replace(tzinfo=None)


def floor_5min(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0, minute=(dt.minute // 5) * 5)


def _to_float(value):
    if value in (None, "", "-"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def fetch_state_demand(state_name: str = DEFAULT_STATE_NAME,
                       state_code: str = DEFAULT_STATE_CODE,
                       timeout: int = 20,
                       when: datetime | None = None) -> MeritReading:
    """Fetch one live demand reading for a state from MERIT.

    Always returns a MeritReading. ``demand_mw`` is None when the feed is idle
    (null) or every endpoint failed; inspect ``ok`` / ``error`` to distinguish.
    """
    ts = floor_5min(when or now_ist())
    last_err = None

    for url in STATE_STATUS_ENDPOINTS:
        try:
            r = requests.get(url, params={"StateName": state_name},
                             headers=HEADERS, timeout=timeout, verify=False)
        except requests.exceptions.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue

        if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
            last_err = f"HTTP {r.status_code} / non-JSON from {url}"
            continue

        try:
            data = r.json()
        except ValueError as exc:
            last_err = f"invalid JSON from {url}: {exc}"
            continue

        row = (data[0] if isinstance(data, list) and data
               else data if isinstance(data, dict) else {})
        return MeritReading(
            state_code=state_code, state_name=state_name, timestamp=ts,
            demand_mw=_to_float(row.get("Demand")),
            isgs_mw=_to_float(row.get("ISGS")),
            import_mw=_to_float(row.get("ImportData")),
            endpoint=url, ok=True,
        )

    return MeritReading(state_code, state_name, ts, None, ok=False,
                        error=last_err or "all endpoints failed")


def fetch_national_demand(timeout: int = 20) -> float | None:
    """Parse 'DEMAND MET <n> MW' from the all-India current table (context only)."""
    try:
        r = requests.get(ALL_INDIA_URL, headers=HEADERS, timeout=timeout, verify=False)
    except requests.exceptions.RequestException:
        return None
    m = re.search(r"DEMAND MET[^0-9]*([\d,]+)", r.text)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def save_reading(reading: MeritReading, dry_run: bool = False) -> str:
    """Upsert a reading into StateLoad5Min keyed on (state, datetime).

    Returns one of: 'skipped_null' | 'dry_run' | 'created' | 'updated'.
    Lazily imports the model so this module stays import-safe without Django.
    """
    if reading.demand_mw is None:
        return "skipped_null"
    if dry_run:
        return "dry_run"

    from power.models import StateLoad5Min  # lazy: keep module Django-optional

    _, created = StateLoad5Min.objects.update_or_create(
        state=reading.state_code,
        datetime=reading.timestamp,
        defaults={"load_mw": reading.demand_mw},
    )
    return "created" if created else "updated"
