"""
APScheduler integration — auto-poll MERIT CG every 5 minutes while the Django
server runs.

Started from ``PowerConfig.ready()`` (power/apps.py). It is guarded so it only
runs inside a real server process:

* ``runserver``  -> only the reloader child (RUN_MAIN=true) or with --noreload,
                    so the dev autoreloader doesn't start two schedulers.
* ``gunicorn`` / ``uvicorn`` -> allowed (one scheduler per worker; writes are
                    idempotent upserts so overlapping workers don't duplicate).
* any other management command (migrate, shell, the poll subprocess, …) -> NOT
                    started.

Toggle with ``MERIT_SCHEDULER_ENABLED`` in settings (default True) or the env var
``MERIT_SCHEDULER`` (``0`` to force off, ``1`` to force on).

Writes go through the same power/utils/merit.py helpers as the management
command, so there is a single source of truth for fetch + save.
"""

from __future__ import annotations

import os
import sys

from power.utils.logger import get_logger

log = get_logger("MeritScheduler")

POLL_INTERVAL_MINUTES = 5
_scheduler = None  # module-level singleton


def poll_once() -> None:
    """One fetch+save cycle (the APScheduler job). Never raises out of the job."""
    try:
        from power.utils.merit import fetch_state_demand, save_reading

        reading = fetch_state_demand()
        ts = reading.timestamp.strftime("%Y-%m-%d %H:%M")
        if reading.demand_mw is None:
            why = "idle/null" if reading.ok else f"error: {reading.error}"
            log.info(f"[{ts}] CG demand {why} — skipped")
            return
        status = save_reading(reading)
        log.info(f"[{ts}] CG demand = {reading.demand_mw:,.0f} MW -> {status}")
    except Exception as exc:  # noqa: BLE001 — a job error must not kill the scheduler
        log.error(f"poll_once failed: {exc}")


def should_autostart() -> bool:
    """Decide whether ready() should start the scheduler in this process."""
    from django.conf import settings

    env = os.environ.get("MERIT_SCHEDULER")
    if env == "0":
        return False
    if env == "1":
        return True
    if not getattr(settings, "MERIT_SCHEDULER_ENABLED", True):
        return False

    argv = sys.argv or []
    subcmd = argv[1] if len(argv) > 1 else ""
    prog = os.path.basename(argv[0]) if argv else ""

    if subcmd == "runserver":
        # only the reloader child actually serves; avoid the parent watcher
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in argv

    # gunicorn/uvicorn (manage.py is not the entrypoint)
    if "gunicorn" in prog or "uvicorn" in prog:
        return True

    return False


def start() -> None:
    """Start the background scheduler exactly once."""
    global _scheduler
    if _scheduler is not None:
        return

    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    _scheduler.add_job(
        poll_once,
        IntervalTrigger(minutes=POLL_INTERVAL_MINUTES),
        id="merit_cg_poll",
        name="Poll MERIT CG demand",
        max_instances=1,     # never overlap a slow poll with the next
        coalesce=True,       # if the process was busy, run once not N times
        replace_existing=True,
    )
    _scheduler.start()
    log.info(f"APScheduler started — poll_merit_cg every {POLL_INTERVAL_MINUTES} min "
             f"(pid {os.getpid()})")

    import atexit
    atexit.register(_shutdown)


def _shutdown() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            pass
        _scheduler = None
