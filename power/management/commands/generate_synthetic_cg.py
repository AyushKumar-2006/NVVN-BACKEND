"""Generate hyper-realistic synthetic CG 5-min demand for 2022-2023.

Learns patterns from the real StateLoad5Min CG data (2024-2026), generates a
back-cast for the missing years with festival/weather/noise/growth overlays,
stages it as state='CG_SYNTH', runs the validation gate (item 10), and — only if
every check passes — promotes the same rows to state='CG'.

    python manage.py generate_synthetic_cg                 # full run + promote
    python manage.py generate_synthetic_cg --analyze-only  # just the pattern report
    python manage.py generate_synthetic_cg --dry-run       # generate+validate, no DB
    python manage.py generate_synthetic_cg --no-promote    # stage CG_SYNTH only

ADDITIVE: never deletes real CG rows. Only CG_SYNTH (its own staging rows) are
cleared for the target range before re-staging, and CG is inserted with
ignore_conflicts strictly inside 2022-2023.
"""

from __future__ import annotations

import os
from datetime import date, datetime

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from power.models import StateLoad5Min
from power.ml.synthetic.generator import (
    GenConfig, generate, validate, compare,
)
from power.ml.synthetic.events import festival_label_map

RAIPUR_LAT, RAIPUR_LON = 21.2514, 81.6296
TMP_DIR = os.path.join(settings.BASE_DIR, "power", "ml", "tmp")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
def _season(m):
    return ("Winter" if m in (12, 1, 2) else "Summer" if m in (3, 4, 5, 6)
            else "Monsoon" if m in (7, 8, 9) else "Spring/Post-monsoon")


def fetch_raipur_temp(start: date, end: date, stdout=None):
    """Hourly 2-m temperature for Raipur from open-meteo archive (CSV-cached)."""
    os.makedirs(TMP_DIR, exist_ok=True)
    cache = os.path.join(TMP_DIR, f"cg_raipur_temp_{start}_{end}.csv")
    if os.path.exists(cache):
        s = pd.read_csv(cache, parse_dates=["time"]).set_index("time")["temperature_c"]
        return s
    import requests
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={"latitude": RAIPUR_LAT, "longitude": RAIPUR_LON,
                "start_date": str(start), "end_date": str(end),
                "hourly": "temperature_2m", "timezone": "Asia/Kolkata"},
        timeout=60,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    s = pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), name="temperature_c")
    s = s.dropna()
    s.rename_axis("time").reset_index().to_csv(cache, index=False)
    if stdout:
        stdout(f"  fetched {len(s)} hourly temps for Raipur -> cached {os.path.basename(cache)}")
    return s


# --------------------------------------------------------------------------- #
class Command(BaseCommand):
    help = "Generate + validate + promote synthetic CG 5-min demand (2022-2023)."

    def add_arguments(self, p):
        p.add_argument("--start", default="2022-01-01")
        p.add_argument("--end", default="2023-12-31")
        p.add_argument("--real-start", default="2024-01-01",
                       help="first date of REAL data; patterns are learned only "
                            "from datetime>=this, and synthesis must end before it")
        p.add_argument("--state", default="CG", help="real source state code")
        p.add_argument("--synth-state", default="CG_SYNTH", help="staging state code")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--growth-2022", type=float, default=0.06)
        p.add_argument("--growth-2023", type=float, default=0.03)
        p.add_argument("--no-weather", action="store_true", help="skip weather correlation")
        p.add_argument("--no-promote", action="store_true", help="stage CG_SYNTH only")
        p.add_argument("--dry-run", action="store_true", help="no DB writes at all")
        p.add_argument("--analyze-only", action="store_true", help="print real-data report and exit")
        p.add_argument("--batch-size", type=int, default=500)

    # ---- helpers -------------------------------------------------------- #
    def _load_real(self, state, real_start):
        df = pd.DataFrame(
            StateLoad5Min.objects.filter(state=state, datetime__gte=real_start)
            .values("datetime", "load_mw").order_by("datetime")
        )
        if df.empty:
            raise CommandError(f"No real data for state={state!r}")
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["load_mw"] = df["load_mw"].astype(float)
        return df

    def _analyze(self, real):
        s = real.set_index("datetime")["load_mw"]
        idx = s.index
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING("\n=== REAL CG PATTERN ANALYSIS (source for synthesis) ==="))
        w(f"rows={len(s):,}  range {idx.min()} -> {idx.max()}")
        am = s.groupby(idx.year).mean()
        w("Annual mean MW : " + "  ".join(f"{y}:{v:,.0f}" for y, v in am.items()))
        yrs = sorted(am.index)
        if len(yrs) >= 2:
            g = (am[yrs[1]] / am[yrs[0]] - 1) * 100
            w(f"  -> measured YoY growth {yrs[0]}->{yrs[1]}: {g:+.1f}%")

        mm = s.groupby(idx.month).mean()
        w("\nSeasonal / month-wise mean MW:")
        for m in range(1, 13):
            w(f"  {MONTHS[m-1]} ({_season(m):<18}): {mm[m]:,.0f}")
        w(f"  peak month: {MONTHS[mm.idxmax()-1]} {mm.max():,.0f} | "
          f"trough: {MONTHS[mm.idxmin()-1]} {mm.min():,.0f}")

        hm = s.groupby(idx.hour).mean()
        w("\nDaily load curve (hourly mean MW):")
        for block in (range(0, 12), range(12, 24)):
            w("  " + " ".join(f"{h:02d}:{hm[h]:,.0f}" for h in block))
        w(f"  morning peak ~h{hm[:12].idxmax():02d} {hm[:12].max():,.0f} | "
          f"evening peak ~h{hm[16:].idxmax():02d} {hm[16:].max():,.0f} | "
          f"overnight base ~{hm[0:5].mean():,.0f}")

        wk = s[idx.dayofweek < 5].mean()
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dm = s.groupby(idx.dayofweek).mean()
        w("\nWeekly pattern (vs weekday mean):")
        w("  " + "  ".join(f"{names[d]}:{(dm[d]/wk-1)*100:+.1f}%" for d in range(7)))
        w(f"  weekend vs weekday: {(s[idx.dayofweek>=5].mean()/wk-1)*100:+.1f}%  "
          "(CG is industrial-baseload-heavy -> small weekend dip)")

        lf = s.mean() / s.max()
        jmax = s.diff().abs().max()
        w(f"\nLoad factor (mean/peak): {lf:.3f}  | peak {s.max():,.0f} MW")
        w(f"5-min jump: max {jmax:,.0f}  p99 {s.diff().abs().quantile(.99):,.0f}  "
          f"mean {s.diff().abs().mean():.1f} MW  (=> noise must be smooth)")

    def _print_validation(self, checks, cfg):
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING("\n=== VALIDATION (state=CG_SYNTH) ==="))
        ok = self.style.SUCCESS("PASS"); bad = self.style.ERROR("FAIL")
        c = checks
        w(f"  [{ok if c['no_negative']['pass'] else bad}] no negatives           "
          f"min={c['no_negative']['min']:.1f} MW")
        w(f"  [{ok if c['max_jump_300']['pass'] else bad}] max 5-min jump <= {cfg.max_jump_mw:.0f} "
          f"   actual max={c['max_jump_300']['max_jump_mw']:.1f} MW")
        lfc = c["load_factor"]
        w(f"  [{ok if lfc['pass'] else bad}] load factor in {lfc['band']}   "
          f"actual={lfc['load_factor']:.3f}")
        mc = c["monthly_within_5pct"]
        w(f"  [{ok if mc['pass'] else bad}] monthly within +/-{cfg.monthly_tol*100:.0f}% of CEA-proxy "
          f"  worst dev={mc['worst_dev_pct']:.2f}% @ {mc['worst_month']}")
        w(f"  ==> OVERALL: {ok if c['all_pass'] else bad}")

    def _print_compare(self, cmp):
        w = self.stdout.write
        w(self.style.MIGRATE_HEADING("\n=== COMPARISON: synthetic vs real distribution ==="))
        ro, so = cmp["overall"]["real"], cmp["overall"]["synth"]
        w(f"{'stat':<14}{'REAL(24-26)':>14}{'SYNTH(22-23)':>14}")
        for k, lab in [("mean", "mean MW"), ("std", "std MW"), ("cov_pct", "CoV %"),
                       ("min", "min"), ("p5", "p5"), ("p50", "median"),
                       ("p95", "p95"), ("max", "max"), ("load_factor", "load factor")]:
            fmt = (lambda x: f"{x:,.3f}") if k in ("load_factor",) else (lambda x: f"{x:,.0f}")
            w(f"{lab:<14}{fmt(ro[k]):>14}{fmt(so[k]):>14}")
        w(f"\nweekend effect : real {cmp['weekend_effect_pct']['real']:+.1f}%   "
          f"synth {cmp['weekend_effect_pct']['synth']:+.1f}%")
        w("\nMonth-wise mean (real climatology vs synth):")
        rm, sm = cmp["monthly"]["real"], cmp["monthly"]["synth"]
        for m in range(1, 13):
            w(f"  {MONTHS[m-1]}: real {rm.get(m,0):,.0f}  synth {sm.get(m,0):,.0f}")

    def _write_rows(self, state, synth, batch_size, replace_range=None):
        """Insert synth rows for `state`. If replace_range given, delete that
        (state, datetime-range) first (used only for the CG_SYNTH staging state)."""
        if replace_range is not None:
            lo, hi = replace_range
            StateLoad5Min.objects.filter(
                state=state, datetime__gte=lo, datetime__lte=hi
            ).delete()
        objs = [StateLoad5Min(state=state, datetime=dt.to_pydatetime(), load_mw=float(v))
                for dt, v in zip(synth["datetime"], synth["load_mw"])]
        with transaction.atomic():
            StateLoad5Min.objects.bulk_create(
                objs, batch_size=batch_size, ignore_conflicts=True
            )
        return len(objs)

    # ---- main ----------------------------------------------------------- #
    def handle(self, *args, **o):
        w = self.stdout.write
        start = datetime.strptime(o["start"], "%Y-%m-%d").date()
        end = datetime.strptime(o["end"], "%Y-%m-%d").date()
        real_start = datetime.strptime(o["real_start"], "%Y-%m-%d").date()

        # safety guard: synthesis must end strictly before the real-data boundary,
        # so promotion can never overwrite real observations (robust to re-runs,
        # even after synthetic rows already live in the CG table).
        if end >= real_start:
            raise CommandError(
                f"end={end} must be before --real-start={real_start}; "
                "refuse to synthesise into the real-data range.")

        # learn ONLY from real data (>= real_start), never from prior synthetic output
        real = self._load_real(o["state"], real_start)
        self._analyze(real)
        if o["analyze_only"]:
            return

        cfg = GenConfig(growth_2022=o["growth_2022"], growth_2023=o["growth_2023"],
                        seed=o["seed"])

        temp = None
        if not o["no_weather"]:
            w(self.style.MIGRATE_HEADING("\n=== WEATHER (real Raipur temperature) ==="))
            try:
                temp = fetch_raipur_temp(start, end, stdout=w)
            except Exception as e:  # noqa: BLE001
                w(self.style.WARNING(f"  weather fetch failed ({e}); "
                                     "continuing without weather correlation"))

        # real load at the real-data boundary -> seam continuity target
        seam_dt = datetime.combine(real_start, datetime.min.time())
        seam_row = (StateLoad5Min.objects.filter(state=o["state"], datetime__gte=seam_dt)
                    .order_by("datetime").values_list("load_mw", flat=True).first())

        w(self.style.MIGRATE_HEADING("\n=== GENERATING ==="))
        synth, meta = generate(real, start, end, cfg, temp_hourly=temp,
                               seam_target=seam_row)
        for yr, gi in meta["growth"].items():
            w(f"  {yr}: target mean {gi['target_mean']:,.0f} MW "
              f"(scale {gi['scale']:.4f} on base {gi['base_pred_mean']:,.0f})")
        flabels = festival_label_map()
        w(f"  rows={meta['n_rows']:,}  festival-shaped 5-min blocks={meta['festival_dates']:,}  "
          f"festival dates={len(flabels)}  weather={'on' if meta['weather_used'] else 'off'}")

        checks = validate(synth, meta["profile"], meta["growth"], cfg)
        self._print_validation(checks, cfg)

        cmp = compare(real, synth)
        self._print_compare(cmp)

        if o["dry_run"]:
            w(self.style.WARNING("\n--dry-run: no rows written."))
            return

        lo = pd.Timestamp(start); hi = pd.Timestamp(end) + pd.Timedelta("23:55:00")
        n_syn = self._write_rows(o["synth_state"], synth, o["batch_size"],
                                 replace_range=(lo, hi))
        w(self.style.SUCCESS(f"\nStaged {n_syn:,} rows as state={o['synth_state']!r}."))

        if not checks["all_pass"]:
            w(self.style.ERROR(
                "Validation FAILED -> NOT promoting to CG. Inspect CG_SYNTH; "
                "re-run after adjusting parameters."))
            return
        if o["no_promote"]:
            w(self.style.WARNING("--no-promote: leaving data as CG_SYNTH only."))
            return

        # delete-then-insert is safe: replace_range (lo..hi) is strictly < real_min
        # (guarded above), so only previously-promoted synthetic CG rows are touched.
        n_cg = self._write_rows(o["state"], synth, o["batch_size"],
                                replace_range=(lo, hi))
        total = StateLoad5Min.objects.filter(
            state=o["state"], datetime__gte=lo, datetime__lte=hi).count()
        w(self.style.SUCCESS(
            f"Validation PASSED -> promoted {n_cg:,} rows to state={o['state']!r} "
            f"(now {total:,} {o['state']} rows in {start}..{end})."))
        w(self.style.SUCCESS(
            f"Extended {o['state']} series: synthetic {start}..{end} + real "
            f"{real_start}.. -> ready for retraining."))
