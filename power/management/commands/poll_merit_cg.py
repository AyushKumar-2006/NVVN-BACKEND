"""
Django management command: poll_merit_cg

Polls MERIT India for Chhattisgarh (CG) live demand and writes each reading to
the StateLoad5Min table (state='CG'). Runs continuously every --interval seconds
until stopped (Ctrl-C / SIGTERM), or a single time with --once.

    python manage.py poll_merit_cg                 # every 5 min, forever
    python manage.py poll_merit_cg --once          # one reading then exit
    python manage.py poll_merit_cg --interval 60   # every minute
    python manage.py poll_merit_cg --dry-run       # fetch + log, no DB write
    python manage.py poll_merit_cg --log-file logs/merit_cg_collection.log

Null handling
-------------
MERIT's live feed is frequently idle and returns Demand=null (no state reporting
at that instant). That is logged as 'idle/null' and skipped — it is NOT an error
and does not stop the loop. Only real numeric readings are written to the DB.

Storage
-------
Each reading is upserted on (state, datetime) where datetime is the current IST
time floored to the 5-min slot, so re-polling within the same slot updates that
row instead of duplicating it (idempotent).

ADDITIVE ONLY — new file; reuses power/utils/merit.py and the StateLoad5Min model.
"""

from __future__ import annotations

import logging
import os
import signal
import time

from django.core.management.base import BaseCommand

from power.utils.logger import get_logger
from power.utils.merit import (
    DEFAULT_STATE_CODE,
    DEFAULT_STATE_NAME,
    fetch_state_demand,
    save_reading,
)


def _attach_file_handler(logger: logging.Logger, path: str) -> None:
    """Append plain (no ANSI colour) log lines to `path` as well as stdout."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == path:
            return  # already attached
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)


class Command(BaseCommand):
    help = ("Poll MERIT India for CG live demand and save to StateLoad5Min. "
            "Runs continuously every --interval seconds until stopped.")

    def add_arguments(self, parser):
        parser.add_argument("--interval", type=int, default=300,
                            help="seconds between polls [default: 300 = 5 min]")
        parser.add_argument("--state", default=DEFAULT_STATE_NAME,
                            help=f"MERIT state name [default: {DEFAULT_STATE_NAME}]")
        parser.add_argument("--state-code", default=DEFAULT_STATE_CODE,
                            help=f"DB state code [default: {DEFAULT_STATE_CODE}]")
        parser.add_argument("--once", action="store_true",
                            help="poll a single time and exit")
        parser.add_argument("--max-polls", type=int, default=0,
                            help="stop after N polls [default: 0 = infinite]")
        parser.add_argument("--timeout", type=int, default=20,
                            help="per-request network timeout (seconds) [default: 20]")
        parser.add_argument("--no-align", action="store_true",
                            help="don't align sleeps to the wall-clock interval boundary")
        parser.add_argument("--dry-run", action="store_true",
                            help="fetch + log but do not write to the database")
        parser.add_argument("--log-file", default=None,
                            help="also append logs to this file")

    # ---------------------------------------------------------------- handle
    def handle(self, *args, **opts):
        log = get_logger("MeritPoll")
        if opts["log_file"]:
            _attach_file_handler(log, opts["log_file"])

        interval = max(5, opts["interval"])
        once = opts["once"]
        max_polls = max(0, opts["max_polls"])
        state = opts["state"]
        state_code = opts["state_code"]
        timeout = opts["timeout"]
        align = not opts["no_align"]
        dry_run = opts["dry_run"]

        self._stop = False

        def _on_signal(signum, _frame):
            log.warning(f"signal {signum} received — stopping after this cycle")
            self._stop = True

        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

        mode = "once" if once else f"every {interval}s"
        log.info(f"poll_merit_cg started — state={state} ({state_code}), {mode}"
                 f"{', DRY-RUN' if dry_run else ''}")

        stats = {"polls": 0, "saved": 0, "updated": 0, "null": 0, "errors": 0}

        try:
            while not self._stop:
                self._poll_once(log, state, state_code, timeout, dry_run, stats)
                stats["polls"] += 1

                if once or (max_polls and stats["polls"] >= max_polls):
                    break
                if self._stop:
                    break
                self._sleep(interval, align)
        finally:
            log.info(
                f"poll_merit_cg stopped — polls={stats['polls']} "
                f"created={stats['saved']} updated={stats['updated']} "
                f"null={stats['null']} errors={stats['errors']}"
            )

    # ---------------------------------------------------------------- helpers
    def _poll_once(self, log, state, state_code, timeout, dry_run, stats):
        try:
            reading = fetch_state_demand(state, state_code, timeout)
        except Exception as exc:  # noqa: BLE001 — never let one bad poll kill the loop
            stats["errors"] += 1
            log.error(f"fetch failed: {exc}")
            return

        ts = reading.timestamp.strftime("%Y-%m-%d %H:%M")

        if reading.demand_mw is None:
            stats["null"] += 1
            if reading.ok:
                log.info(f"[{ts}] {state_code} demand = idle/null "
                         f"(feed not reporting) — skipped")
            else:
                stats["errors"] += 1
                log.warning(f"[{ts}] {state_code} fetch error: {reading.error} — skipped")
            return

        status = save_reading(reading, dry_run=dry_run)
        if status == "created":
            stats["saved"] += 1
        elif status == "updated":
            stats["updated"] += 1

        extra = []
        if reading.isgs_mw is not None:
            extra.append(f"ISGS={reading.isgs_mw:.0f}")
        if reading.import_mw is not None:
            extra.append(f"import={reading.import_mw:.0f}")
        suffix = ("  " + " ".join(extra)) if extra else ""
        log.info(f"[{ts}] {state_code} demand = {reading.demand_mw:,.0f} MW "
                 f"-> {status}{suffix}")

    def _sleep(self, interval, align):
        """Sleep `interval` seconds (or until the next wall-clock boundary when
        aligned), waking promptly if a stop signal arrives."""
        if align:
            remaining = interval - (time.time() % interval)
        else:
            remaining = interval
        deadline = time.time() + remaining
        while not self._stop and time.time() < deadline:
            time.sleep(min(1.0, deadline - time.time()))
