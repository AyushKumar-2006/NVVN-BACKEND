"""Django management command: backfill_cg_live

Brings the CG series up to *today* so the live dashboard never shows a gap:

  1. LOAD  — fills the CG 5-min series (state='CG') from the last stored point
             up to today, using the existing synthetic generator
             (power/ml/synthetic/generator.py). New rows only — existing rows are
             never overwritten (bulk_create ignore_conflicts), and the head is
             seam-blended to the last real value so the join is continuous.
  2. TEMP  — extends per-district + population-weighted daily temperature into
             power_weather from where CG weather ends up to today+`--forecast-days`,
             using the same 4 districts/weights as power/ml/weather.py (open-meteo
             archive for past days, forecast for today..+15, 30-year climate normal
             — i.e. prior-year seasonal pattern — beyond).

    python manage.py backfill_cg_live
    python manage.py backfill_cg_live --today 2026-06-07 --forecast-days 30
    python manage.py backfill_cg_live --skip-weather
    python manage.py backfill_cg_live --dry-run

Idempotent: safe to re-run daily (e.g. from scripts/fetch_daily.sh).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from power.models import StateLoad5Min, Weather

CG = "CG"
REAL_START = date(2024, 1, 1)        # learn the profile from real data only


class Command(BaseCommand):
    help = "Fill CG 5-min load + per-district weather up to today (idempotent)."

    def add_arguments(self, p):
        p.add_argument("--today", default=None, help="override 'today' (YYYY-MM-DD)")
        p.add_argument("--forecast-days", type=int, default=30,
                       help="extend weather this many days past today [30]")
        p.add_argument("--skip-load", action="store_true")
        p.add_argument("--skip-weather", action="store_true")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--batch-size", type=int, default=500)

    # --------------------------------------------------------------- handle
    def handle(self, *args, **o):
        today = (datetime.strptime(o["today"], "%Y-%m-%d").date()
                 if o["today"] else datetime.now().date())
        self.dry = o["dry_run"]
        self.bs = o["batch_size"]
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING(f"\n=== backfill_cg_live (today={today}"
                                     f"{' DRY-RUN' if self.dry else ''}) ==="))
        if not o["skip_load"]:
            self._fill_load(today, w)
        if not o["skip_weather"]:
            self._fill_weather(today, o["forecast_days"], w)
        w(self.style.SUCCESS("\nDone."))

    # --------------------------------------------------------------- load
    def _fill_load(self, today, w):
        from power.ml.synthetic.generator import GenConfig, generate

        last = (StateLoad5Min.objects.filter(state=CG)
                .order_by("-datetime").values_list("datetime", "load_mw").first())
        if not last:
            raise CommandError("No existing CG 5-min data to learn from.")
        last_dt, last_val = last
        start = (last_dt.date() + timedelta(days=1))
        end = today
        w(self.style.HTTP_INFO(
            f"\n[1/2] LOAD: CG 5-min  last={last_dt}  ->  fill {start}..{end}"))
        if start > end:
            w("   already current — nothing to fill."); return

        real = pd.DataFrame(
            StateLoad5Min.objects.filter(state=CG, datetime__gte=REAL_START)
            .order_by("datetime").values("datetime", "load_mw"))
        real["datetime"] = pd.to_datetime(real["datetime"])
        real["load_mw"] = real["load_mw"].astype(float)

        synth, meta = generate(real, start, end, GenConfig())

        # seam-blend the head toward the last real value (continuous join)
        load = synth["load_mw"].to_numpy(dtype=float).copy()
        k = min(24, load.size)
        if k > 0 and last_val is not None:
            wgt = np.linspace(1.0, 0.0, k)            # 1 at the seam, fading out
            load[:k] = load[:k] * (1 - wgt) + float(last_val) * wgt
        synth["load_mw"] = np.round(load, 1)

        rows = [StateLoad5Min(state=CG, datetime=dt.to_pydatetime(),
                              load_mw=float(v))
                for dt, v in zip(synth["datetime"], synth["load_mw"])]
        w(f"   generated {len(rows):,} 5-min rows  "
          f"(mean {synth['load_mw'].mean():,.0f} MW, "
          f"peak {synth['load_mw'].max():,.0f} MW)")
        if self.dry:
            w(self.style.WARNING("   DRY-RUN: not written.")); return
        created = StateLoad5Min.objects.bulk_create(
            rows, batch_size=self.bs, ignore_conflicts=True)
        w(self.style.SUCCESS(
            f"   inserted into state='CG' (ignore_conflicts; existing rows kept)."))

    # --------------------------------------------------------------- weather
    def _fill_weather(self, today, fc_days, w):
        from power.ml import weather as W

        wx_last = (Weather.objects.filter(state=CG)
                   .order_by("-datetime").values_list("datetime", flat=True).first())
        gap_start = (wx_last.date() + timedelta(days=1)) if wx_last else date(2025, 6, 1)
        end = today + timedelta(days=fc_days)
        total_days = (end - gap_start).days + 1
        w(self.style.HTTP_INFO(
            f"\n[2/2] TEMP: power_weather  last={wx_last}  ->  fill "
            f"{gap_start}..{end}  ({total_days} days x 4 districts)"))
        if total_days <= 0:
            w("   already current — nothing to fill."); return

        try:
            data = W.fetch_daily_weather_districts(gap_start, total_days)
        except Exception as e:  # noqa: BLE001
            w(self.style.WARNING(f"   weather fetch failed ({e}); skipping temp fill."))
            return

        objs = []
        # per-district rows (state='CG_<District>')
        for d in data["districts"]:
            code = f"CG_{d['name']}"
            for pt in d["series"]:
                t = pt["temperature_c"]
                if t is None:
                    continue
                objs.append(Weather(
                    state=code, datetime=datetime.combine(date.fromisoformat(pt["date"]), time.min),
                    frequency="daily", temperature_c=float(t), source="open-meteo"))
        # weighted blend (state='CG')
        filled = 0
        for pt in data["weighted"]:
            t = pt["temperature_c"]
            if t is None:
                continue
            filled += 1
            objs.append(Weather(
                state=CG, datetime=datetime.combine(date.fromisoformat(pt["date"]), time.min),
                frequency="daily", temperature_c=float(t), source="open-meteo"))

        names = ", ".join(f"{d['name']}({d['weight']})" for d in data["districts"])
        w(f"   districts: {names}")
        w(f"   built {len(objs):,} daily rows  (weighted days filled: {filled})")
        if self.dry:
            w(self.style.WARNING("   DRY-RUN: not written.")); return
        Weather.objects.bulk_create(objs, batch_size=self.bs, ignore_conflicts=True)
        w(self.style.SUCCESS("   inserted into power_weather (ignore_conflicts)."))
