"""Django management command: setup_cron

Prints the exact crontab line(s) to schedule the daily WRLDC PSP refresh
(`scripts/fetch_daily.sh`) for THIS machine — using the real, absolute project
path and the Python interpreter currently running, so the printed line can be
pasted straight into `crontab -e`.

    python manage.py setup_cron                 # print the crontab line
    python manage.py setup_cron --hour 3        # run at 03:00 instead of 02:00
    python manage.py setup_cron --install       # append it to the user crontab

ADDITIVE: prints only by default; --install is the only path that mutates the
crontab, and it refuses to add a duplicate line.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

MARKER = "# NVVN fetch_daily (WRLDC PSP -> CG_WRLDC)"


class Command(BaseCommand):
    help = "Print (or install) the crontab line for the daily WRLDC PSP fetch on this machine."

    def add_arguments(self, parser):
        parser.add_argument("--hour", type=int, default=2,
                            help="hour of day to run (0-23) [default: 2]")
        parser.add_argument("--minute", type=int, default=0,
                            help="minute of hour to run (0-59) [default: 0]")
        parser.add_argument("--install", action="store_true",
                            help="append the line to the current user crontab "
                                 "(skips if an identical NVVN line already exists)")

    def handle(self, *args, **opts):
        hour = opts["hour"]
        minute = opts["minute"]
        if not (0 <= hour <= 23) or not (0 <= minute <= 59):
            raise CommandError("--hour must be 0-23 and --minute 0-59")

        project_dir = Path(settings.BASE_DIR).resolve()
        script = project_dir / "scripts" / "fetch_daily.sh"
        if not script.exists():
            self.stdout.write(self.style.WARNING(
                f"  note: {script} does not exist yet (expected scripts/fetch_daily.sh)."))

        # Pass the interpreter currently running manage.py through to the script,
        # so cron uses the same environment even with conda / no venv on PATH.
        python_exe = Path(sys.executable).resolve()
        cron_cmd = (
            f"NVVN_PYTHON={python_exe} bash {script}"
        )
        cron_line = f"{minute} {hour} * * * {cron_cmd}"

        w = self.stdout.write
        w(self.style.MIGRATE_HEADING("\n=== Crontab line for this machine ==="))
        w(f"  project : {project_dir}")
        w(f"  python  : {python_exe}")
        w(f"  schedule: every day at {hour:02d}:{minute:02d}")
        w("")
        w(self.style.SUCCESS("  " + MARKER))
        w(self.style.SUCCESS("  " + cron_line))
        w("")
        w("To install it manually, run `crontab -e` and paste the two lines above,")
        w("or run:  python manage.py setup_cron --install")
        w("")
        w("Simpler portable form (uses venv/conda auto-detect inside the script):")
        w(self.style.HTTP_INFO(f"  {minute} {hour} * * * bash {script}"))
        w("")

        if opts["install"]:
            self._install(cron_line, w)

    # ------------------------------------------------------------------ install
    def _install(self, cron_line, w):
        try:
            existing = subprocess.run(
                ["crontab", "-l"], capture_output=True, text=True
            )
        except FileNotFoundError:
            raise CommandError("`crontab` not found on this system; cannot --install.")

        current = existing.stdout if existing.returncode == 0 else ""

        if cron_line in current or MARKER in current:
            w(self.style.WARNING("  --install: an NVVN fetch_daily line already exists; "
                                 "leaving the crontab unchanged."))
            return

        new_crontab = current
        if new_crontab and not new_crontab.endswith("\n"):
            new_crontab += "\n"
        new_crontab += f"{MARKER}\n{cron_line}\n"

        proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True)
        if proc.returncode == 0:
            w(self.style.SUCCESS("  --install: crontab updated. Verify with `crontab -l`."))
        else:
            raise CommandError("failed to write crontab (crontab - returned nonzero).")
