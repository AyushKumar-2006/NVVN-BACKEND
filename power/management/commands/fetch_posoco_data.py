"""
Django management command: fetch_posoco_data

Auto-downloads POSOCO / Grid-India (NLDC) **daily PSP report** PDFs over a date
range, parses the Chhattisgarh (CG) daily figures out of each PDF, and upserts
the energy series into the existing ``StateDailyLoad`` table.

    python manage.py fetch_posoco_data --from 2015-01-01 --to 2026-06-04

What it does
------------
1. Auto-downloads every day's PSP PDF from grid-india.in (cached on disk, so
   re-runs resume instantly and skip already-fetched days).
2. Parses the Chhattisgarh row (peak MW + energy MU) from each PDF.
3. Saves the energy figure to the DB automatically (idempotent upsert).
4. Prints running progress — how many PDFs were downloaded / parsed / missing.

Verified working URL pattern (probed live, works 2013 -> present):
    https://report.grid-india.in/ReportData/Daily Report/PSP Report/
        {FY}/{Month} {YYYY}/{DD.MM.YY}_NLDC_PSP.pdf
    FY = Indian financial-year folder (April-March), e.g. "2024-2025"

Note on availability: the site publishes with a lag, so the most recent days in
the requested range will simply 404 (counted as "missing") until Grid-India
uploads them. That is expected and harmless — re-run later to backfill.

ADDITIVE ONLY — this is a new file. It reuses the verified downloader/parser in
``power/utils/grid_india_psp.py`` and the existing upsert helper in
``power/utils/upload.py``. It does not modify any existing file. ``StateDailyLoad``
has only an ``energy_mu`` column (no peak field), so — consistent with the
existing schema — the energy figure is stored and the parsed peak is reported in
the progress log but not persisted.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import requests
from django.core.management.base import BaseCommand, CommandError

from power.models import StateDailyLoad
from power.utils.upload import bulk_upsert_state_daily
from power.utils.grid_india_psp import HEADERS, STATE_CODE, extract_cg, psp_url

requests.packages.urllib3.disable_warnings()  # gov TLS cert -> verify=False


class Command(BaseCommand):
    help = (
        "Auto-download Grid-India (POSOCO) daily PSP PDFs over a date range, "
        "parse Chhattisgarh (CG) load, and save the energy series to "
        "StateDailyLoad. Example: manage.py fetch_posoco_data "
        "--from 2015-01-01 --to 2026-06-04"
    )

    # ------------------------------------------------------------------ args
    def add_arguments(self, parser):
        parser.add_argument(
            "--from", dest="dfrom", required=True,
            help="start date (inclusive), YYYY-MM-DD",
        )
        parser.add_argument(
            "--to", dest="dto", default=date.today().isoformat(),
            help="end date (inclusive), YYYY-MM-DD [default: today]",
        )
        parser.add_argument(
            "--cache-dir", default="/tmp/psp_cache",
            help="directory PDFs are downloaded/cached into [default: /tmp/psp_cache]",
        )
        parser.add_argument(
            "--batch-size", type=int, default=200,
            help="flush parsed rows to the DB every N records [default: 200]",
        )
        parser.add_argument(
            "--progress-every", type=int, default=25,
            help="print a progress line every N days [default: 25]",
        )
        parser.add_argument(
            "--sleep", type=float, default=0.2,
            help="seconds to pause after each fresh download (be polite) [default: 0.2]",
        )
        parser.add_argument(
            "--retries", type=int, default=3,
            help="network retries per PDF before giving up on a day [default: 3]",
        )
        parser.add_argument(
            "--refresh", action="store_true",
            help="re-download PDFs even if a cached copy exists",
        )
        parser.add_argument(
            "--dry-run", action="store_true",
            help="download + parse but do not write anything to the database",
        )

    # ---------------------------------------------------------------- handle
    def handle(self, *args, **opts):
        try:
            start = date.fromisoformat(opts["dfrom"])
            end = date.fromisoformat(opts["dto"])
        except ValueError as exc:
            raise CommandError(f"invalid date (use YYYY-MM-DD): {exc}")
        if start > end:
            raise CommandError("--from must be on or before --to")

        cache_dir = opts["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)

        batch_size = max(1, opts["batch_size"])
        progress_every = max(1, opts["progress_every"])
        sleep = max(0.0, opts["sleep"])
        retries = max(1, opts["retries"])
        refresh = opts["refresh"]
        dry_run = opts["dry_run"]

        total_days = (end - start).days + 1
        self.stdout.write(self.style.MIGRATE_HEADING(
            f"Fetching POSOCO/Grid-India daily PSP for CG: "
            f"{start} -> {end}  ({total_days} day(s))"
        ))
        self.stdout.write(f"  source : {psp_url(start)}")
        self.stdout.write(f"  cache  : {cache_dir}"
                          f"{'   (--refresh: ignoring cache)' if refresh else ''}")
        if dry_run:
            self.stdout.write(self.style.WARNING("  DRY-RUN: nothing will be written to the DB"))
        self.stdout.write("")

        stats = {
            "downloaded": 0, "cached": 0, "missing": 0, "error": 0,
            "parsed": 0, "no_cg": 0, "saved": 0,
        }
        buffer: list[StateDailyLoad] = []

        def flush():
            if buffer and not dry_run:
                bulk_upsert_state_daily(buffer)
                stats["saved"] += len(buffer)
            buffer.clear()

        cur = start
        i = 0
        try:
            while cur <= end:
                i += 1
                path, status = self._download(cur, cache_dir, retries, refresh)
                stats[status] += 1

                peak = energy = None
                if path:
                    res = self._parse(path)
                    if res and res.get("energy_mu") is not None:
                        peak, energy = res.get("peak_mw"), res["energy_mu"]
                        buffer.append(StateDailyLoad(
                            state=STATE_CODE, date=cur, energy_mu=float(energy),
                        ))
                        stats["parsed"] += 1
                    else:
                        stats["no_cg"] += 1
                    if status == "downloaded" and sleep:
                        time.sleep(sleep)

                if len(buffer) >= batch_size:
                    flush()

                if i % progress_every == 0 or cur == end:
                    self._progress(i, total_days, cur, stats, peak, energy)

                cur += timedelta(days=1)
        finally:
            flush()  # persist whatever we have, even on Ctrl-C

        self._summary(stats, dry_run)

    # --------------------------------------------------------------- helpers
    def _download(self, d: date, cache_dir: str, retries: int, refresh: bool):
        """Return (local_path|None, status).

        status: 'downloaded' | 'cached' | 'missing' (404) | 'error' (network).
        PDFs are cached by date so re-runs skip already-fetched days.
        """
        path = os.path.join(cache_dir, f"PSP_{d.isoformat()}.pdf")
        if not refresh and os.path.exists(path) and os.path.getsize(path) > 1000:
            return path, "cached"

        url = psp_url(d)
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, headers=HEADERS, timeout=40, verify=False)
                if r.status_code == 200 and r.content[:4] == b"%PDF":
                    with open(path, "wb") as fh:
                        fh.write(r.content)
                    return path, "downloaded"
                if r.status_code == 404:
                    return None, "missing"  # not published (yet) for this day
                # other status (5xx, throttling, ...) -> retry
            except requests.RequestException:
                pass
            if attempt < retries:
                time.sleep(1.0 * attempt)  # simple linear backoff
        return None, "error"

    def _parse(self, path: str):
        """Extract the CG row; never let a single bad PDF abort the run."""
        try:
            return extract_cg(path)
        except Exception as exc:  # noqa: BLE001
            self.stderr.write(self.style.WARNING(
                f"  parse failed for {os.path.basename(path)}: {exc}"
            ))
            return None

    def _progress(self, i, total, cur, stats, peak, energy):
        pdfs = stats["downloaded"] + stats["cached"]
        tail = ""
        if energy is not None:
            tail = f"  [{cur}: peak={peak} MW, energy={energy} MU]"
        self.stdout.write(
            f"[{i}/{total}] {cur}  "
            f"PDFs={pdfs} (new {stats['downloaded']}, cached {stats['cached']})  "
            f"parsed={stats['parsed']}  missing={stats['missing']}  "
            f"err={stats['error']}  saved={stats['saved']}{tail}"
        )

    def _summary(self, stats, dry_run):
        pdfs = stats["downloaded"] + stats["cached"]
        self.stdout.write("")
        self.stdout.write(self.style.MIGRATE_HEADING("Done."))
        self.stdout.write(f"  PDFs available   : {pdfs} "
                          f"(downloaded {stats['downloaded']}, from cache {stats['cached']})")
        self.stdout.write(f"  CG rows parsed   : {stats['parsed']}")
        self.stdout.write(f"  missing (404)    : {stats['missing']}   "
                          f"(not published yet / no report that day)")
        self.stdout.write(f"  network errors   : {stats['error']}")
        self.stdout.write(f"  PDFs w/o CG row  : {stats['no_cg']}")
        if dry_run:
            self.stdout.write(self.style.WARNING(
                f"  DRY-RUN: would have upserted {stats['parsed']} StateDailyLoad row(s)"
            ))
        else:
            self.stdout.write(self.style.SUCCESS(
                f"  saved to DB      : {stats['saved']} StateDailyLoad ({STATE_CODE}) row(s)"
            ))
