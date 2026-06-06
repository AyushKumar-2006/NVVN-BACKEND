"""
Django management command: start_data_collection

Starts the MERIT CG poller (``poll_merit_cg``) in the background, detached from
the terminal, logging everything to ``logs/merit_cg_collection.log``. A PID file
(``logs/merit_cg_collection.pid``) prevents a second copy from starting and lets
you stop / check it.

    python manage.py start_data_collection              # start (if not already running)
    python manage.py start_data_collection --status     # is it running? recent log tail
    python manage.py start_data_collection --stop        # stop it
    python manage.py start_data_collection --restart     # stop then start
    python manage.py start_data_collection --foreground  # run inline (no detach)
    python manage.py start_data_collection --interval 60 # poll cadence (seconds)

The background worker is the same ``poll_merit_cg`` command, so it benefits from
the same null-handling, idempotent upserts, and logging.

ADDITIVE ONLY — new file. Writes only under ``logs/`` (git-ignored).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = ("Start/stop/inspect the background MERIT CG poller. "
            "Logs to logs/merit_cg_collection.log.")

    def add_arguments(self, parser):
        parser.add_argument("--interval", type=int, default=300,
                            help="poll cadence in seconds [default: 300 = 5 min]")
        parser.add_argument("--stop", action="store_true", help="stop the running poller")
        parser.add_argument("--status", action="store_true", help="show running status + log tail")
        parser.add_argument("--restart", action="store_true", help="stop then start")
        parser.add_argument("--foreground", action="store_true",
                            help="run the poller inline in this process (no detach)")

    # ---------------------------------------------------------------- handle
    def handle(self, *args, **opts):
        base = Path(settings.BASE_DIR)
        logs_dir = base / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "merit_cg_collection.log"
        pid_file = logs_dir / "merit_cg_collection.pid"

        if opts["status"]:
            return self._status(pid_file, log_file)

        if opts["stop"]:
            return self._stop(pid_file)

        if opts["restart"]:
            self._stop(pid_file)
            time.sleep(1)

        if opts["foreground"]:
            self.stdout.write(self.style.MIGRATE_HEADING(
                f"Running poll_merit_cg in the foreground (log -> {log_file})"))
            call_command("poll_merit_cg", interval=opts["interval"], log_file=str(log_file))
            return

        self._start(base, log_file, pid_file, opts["interval"])

    # ---------------------------------------------------------------- start
    def _start(self, base, log_file, pid_file, interval):
        pid = self._read_pid(pid_file)
        if pid and self._alive(pid):
            self.stdout.write(self.style.WARNING(
                f"already running (pid {pid}). Use --restart or --stop."))
            return

        cmd = [
            sys.executable, str(base / "manage.py"), "poll_merit_cg",
            "--interval", str(interval), "--log-file", str(log_file),
        ]
        # MERIT_SCHEDULER=0 so the child never also starts the in-process scheduler
        env = {**os.environ, "MERIT_SCHEDULER": "0"}

        # The child writes clean, formatted lines to log_file itself via its
        # --log-file FileHandler. We send its stdout (the colour console handler)
        # to /dev/null so lines aren't duplicated, but keep stderr in the log so
        # an early crash/traceback is still captured.
        log_fh = open(log_file, "a")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=log_fh,
            cwd=str(base), start_new_session=True, env=env,
        )
        pid_file.write_text(str(proc.pid))

        # give it a beat and confirm it didn't immediately die
        time.sleep(1.5)
        if not self._alive(proc.pid):
            self.stdout.write(self.style.ERROR(
                f"poller exited immediately — check {log_file}"))
            return

        self.stdout.write(self.style.SUCCESS(
            f"started background MERIT CG poller (pid {proc.pid}, every {interval}s)"))
        self.stdout.write(f"  log  : {log_file}")
        self.stdout.write(f"  pid  : {pid_file}")
        self.stdout.write(f"  stop : python manage.py start_data_collection --stop")

    # ---------------------------------------------------------------- stop
    def _stop(self, pid_file):
        pid = self._read_pid(pid_file)
        if not pid:
            self.stdout.write("no PID file — poller does not appear to be running.")
            return
        if not self._alive(pid):
            self.stdout.write(f"pid {pid} not running; clearing stale PID file.")
            pid_file.unlink(missing_ok=True)
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            self.stdout.write(self.style.ERROR(f"could not signal pid {pid}: {exc}"))
            return
        # wait for graceful exit
        for _ in range(20):
            if not self._alive(pid):
                break
            time.sleep(0.25)
        if self._alive(pid):
            os.kill(pid, signal.SIGKILL)
        pid_file.unlink(missing_ok=True)
        self.stdout.write(self.style.SUCCESS(f"stopped MERIT CG poller (pid {pid})."))

    # ---------------------------------------------------------------- status
    def _status(self, pid_file, log_file):
        pid = self._read_pid(pid_file)
        if pid and self._alive(pid):
            self.stdout.write(self.style.SUCCESS(f"RUNNING (pid {pid})"))
        else:
            self.stdout.write(self.style.WARNING("NOT running"))
            if pid:
                self.stdout.write(f"  (stale PID file points to {pid})")
        self.stdout.write(f"  log : {log_file}")
        if log_file.exists():
            tail = log_file.read_text(errors="replace").splitlines()[-8:]
            self.stdout.write("  last log lines:")
            for line in tail:
                self.stdout.write(f"    {line}")

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _read_pid(pid_file):
        try:
            return int(pid_file.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None

    @staticmethod
    def _alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
