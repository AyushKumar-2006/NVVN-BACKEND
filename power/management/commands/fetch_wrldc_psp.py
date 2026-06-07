"""
Django management command: fetch_wrldc_psp

Crawls the WRLDC reporting portal's daily PSP Excel archive, downloads every
``WRLDC_PSP_Report_DD-MM-YYYY.xls`` for the requested year(s), parses the real
Chhattisgarh demand points out of each, and upserts them into ``StateLoad5Min``.

    # priority years first
    python manage.py fetch_wrldc_psp --years 2022,2023
    # then the rest
    python manage.py fetch_wrldc_psp --years 2024,2025,2026,2019,2020,2021
    # everything the server has, in one go
    python manage.py fetch_wrldc_psp --all

Source (see power/utils/wrldc_psp.py for the full reverse-engineering notes):
    https://reporting.wrldc.in:8081/PSPExcel/{YEAR}/{Month}/WRLDC_PSP_Report_{DD-MM-YYYY}.xls

What it does
------------
1. Lists each year/month folder (IIS autoindex) to get exact filenames.
2. Downloads each .xls to ``--download-dir`` (default ``~/Downloads/wrldc_data``),
   verifying every file is a complete, openable workbook and retrying the
   server's frequent truncated deliveries. Already-downloaded valid files are
   reused and NEVER deleted/overwritten (safe to resume / re-run).
3. Parses the Chhattisgarh off-peak (03:00), evening-peak (19:00), and
   maximum-demand-of-day (MW + clock time) figures — the only real demand the
   daily report carries (it has no 5-min curve).
4. Upserts those sparse points into ``StateLoad5Min`` under ``--state``
   (default ``CG_WRLDC``, kept separate from the synthetic ``CG`` series) using
   a conflict-update so existing rows are corrected, not duplicated, and no
   range is deleted.

ADDITIVE ONLY — new file; reuses power/utils/wrldc_psp.py and the StateLoad5Min
model. Nothing existing is modified.
"""
from __future__ import annotations

import os
import time
from datetime import date

import requests
from django.core.management.base import BaseCommand, CommandError

from power.models import StateLoad5Min
from power.utils import wrldc_psp as W


class Command(BaseCommand):
    help = (
        "Download WRLDC daily PSP .xls reports and upsert real Chhattisgarh "
        "demand points into StateLoad5Min. Example: "
        "manage.py fetch_wrldc_psp --years 2022,2023"
    )

    def add_arguments(self, parser):
        g = parser.add_mutually_exclusive_group(required=True)
        g.add_argument("--years", help="comma-separated years, e.g. 2022,2023")
        g.add_argument("--all", action="store_true",
                       help="every year the server publishes (2019..present)")
        parser.add_argument(
            "--download-dir", default=os.path.expanduser("~/Downloads/wrldc_data"),
            help="where raw .xls files are saved [default: ~/Downloads/wrldc_data]",
        )
        parser.add_argument(
            "--state", default=W.STATE_CODE,
            help=f"state code to store under [default: {W.STATE_CODE}]",
        )
        parser.add_argument("--limit", type=int, default=0,
                            help="stop after N files per year (testing)")
        parser.add_argument("--retries", type=int, default=4,
                            help="download retries per file [default: 4]")
        parser.add_argument("--sleep", type=float, default=0.15,
                            help="pause after each fresh download [default: 0.15s]")
        parser.add_argument("--progress-every", type=int, default=25,
                            help="print a progress line every N files [default: 25]")
        parser.add_argument("--dry-run", action="store_true",
                            help="download + parse but do not write to the DB")

    # ---------------------------------------------------------------- handle
    def handle(self, *args, **opts):
        session = requests.Session()

        if opts["all"]:
            years = W.list_years(session)
            if not years:
                raise CommandError("could not list years from the WRLDC server")
        else:
            try:
                years = [int(y) for y in opts["years"].split(",") if y.strip()]
            except ValueError:
                raise CommandError("--years must be comma-separated integers")

        dl_dir = os.path.expanduser(opts["download_dir"])
        state = opts["state"]
        limit = opts["limit"]
        retries = max(1, opts["retries"])
        sleep = max(0.0, opts["sleep"])
        every = max(1, opts["progress_every"])
        dry = opts["dry_run"]

        os.makedirs(dl_dir, exist_ok=True)

        self.stdout.write(self.style.MIGRATE_HEADING(
            f"WRLDC PSP -> StateLoad5Min(state='{state}')   years={years}"
        ))
        self.stdout.write(f"  source : {W.BASE}/<year>/<month>/WRLDC_PSP_Report_*.xls")
        self.stdout.write(f"  raw -> : {dl_dir}")
        if dry:
            self.stdout.write(self.style.WARNING("  DRY-RUN: nothing will be written to the DB"))
        self.stdout.write("")

        grand = {k: 0 for k in
                 ("files", "downloaded", "cached", "missing", "corrupt",
                  "error", "parsed", "no_data", "points", "saved")}
        buffer: list[StateLoad5Min] = []

        def flush():
            if buffer and not dry:
                StateLoad5Min.objects.bulk_create(
                    buffer, batch_size=400, update_conflicts=True,
                    unique_fields=["state", "datetime"], update_fields=["load_mw"],
                )
                grand["saved"] += len(buffer)
            buffer.clear()

        for year in years:
            self.stdout.write(self.style.HTTP_INFO(f"== {year}: listing files…"))
            files = W.list_year_files(year, session)
            if limit:
                files = files[:limit]
            self.stdout.write(f"   {len(files)} report file(s) found")

            for i, (d, url) in enumerate(files, 1):
                grand["files"] += 1
                dest = os.path.join(dl_dir, str(d.year),
                                    W.month_name(d.month), os.path.basename(url))
                status = W.download(url, dest, session, retries=retries)
                grand[status] = grand.get(status, 0) + 1

                if status in ("downloaded", "cached"):
                    pts = W.extract_cg_points(dest, d)
                    if pts:
                        grand["parsed"] += 1
                        grand["points"] += len(pts)
                        for p in pts:
                            buffer.append(StateLoad5Min(
                                state=state, datetime=p["datetime"],
                                load_mw=p["load_mw"],
                            ))
                    else:
                        grand["no_data"] += 1
                    if status == "downloaded" and sleep:
                        time.sleep(sleep)

                if len(buffer) >= 400:
                    flush()

                if i % every == 0 or i == len(files):
                    self.stdout.write(
                        f"   [{year} {i}/{len(files)}] {d}  "
                        f"new={grand['downloaded']} cached={grand['cached']} "
                        f"miss={grand['missing']} corrupt={grand['corrupt']} "
                        f"err={grand['error']}  pts={grand['points']} "
                        f"saved={grand['saved']}"
                    )
            flush()

        flush()
        self._summary(grand, state, dry)

    # --------------------------------------------------------------- summary
    def _summary(self, g, state, dry):
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Done."))
        self.stdout.write(f"  files seen        : {g['files']}")
        self.stdout.write(f"  downloaded / cached: {g['downloaded']} / {g['cached']}")
        self.stdout.write(f"  missing (404)     : {g['missing']}")
        self.stdout.write(f"  corrupt/truncated : {g['corrupt']}   (server-side, after retries)")
        self.stdout.write(f"  network errors    : {g['error']}")
        self.stdout.write(f"  files parsed (CG) : {g['parsed']}   (no CG row: {g['no_data']})")
        self.stdout.write(f"  CG demand points  : {g['points']}")
        if dry:
            self.stdout.write(self.style.WARNING(
                f"  DRY-RUN: would have upserted {g['points']} StateLoad5Min row(s)"))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"  upserted to DB    : {g['saved']} StateLoad5Min(state='{state}') row(s)"))
