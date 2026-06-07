"""Django management command: retrain_cg_models

Retrains the Chhattisgarh daily peak-demand models on ALL real
``StateLoad5Min (state='CG_WRLDC')`` data and overwrites the saved artifacts.

    python manage.py retrain_cg_models
    python manage.py retrain_cg_models --test-frac 0.25

Pipeline (see power/ml/cg_forecast.py):
  1. Load all CG_WRLDC rows ordered by datetime -> daily peak series.
  2. Feature engineer: hour, dayofweek, month, is_weekend, lag_1d, lag_7d,
     rolling_mean_7d.
  3. Chronological 80/20 train/test split (time series -> no shuffle).
  4. Train XGBoost and Prophet on the train split.
  5. Print MAE / RMSE / MAPE for both on the test split.
  6. Save both models to power/ml/models/ (cg_xgb.joblib, cg_prophet.joblib),
     overwriting any existing artifacts, and write power/ml/models/eval_metrics.json.

ADDITIVE: new command; only overwrites its own artifacts (cg_xgb.joblib,
cg_prophet.joblib, eval_metrics.json). Never touches the DB or other models.
"""
from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandError

from power.ml import cg_forecast as cg


class Command(BaseCommand):
    help = ("Retrain the CG (Chhattisgarh) XGBoost + Prophet daily peak-demand "
            "models on real CG_WRLDC data and save them with eval metrics.")

    def add_arguments(self, parser):
        parser.add_argument("--test-frac", type=float, default=0.2,
                            help="fraction of most-recent data held out for test [0.2]")
        parser.add_argument("--quiet-prophet", action="store_true", default=True,
                            help="suppress Prophet/cmdstanpy console spam [on]")

    def handle(self, *args, **opts):
        # quiet the very chatty Prophet / cmdstanpy loggers
        for name in ("prophet", "cmdstanpy", "fbprophet"):
            logging.getLogger(name).setLevel(logging.WARNING)

        test_frac = opts["test_frac"]
        if not (0.05 <= test_frac <= 0.5):
            raise CommandError("--test-frac must be between 0.05 and 0.5")

        w = self.stdout.write
        w(self.style.MIGRATE_HEADING(
            f"\n=== Retrain CG models (state='{cg.STATE}', target={cg.TARGET_DESC}) ==="))

        total = cg.data_count()
        w(f"  CG_WRLDC rows in DB : {total:,}")
        if total == 0:
            raise CommandError(
                f"No rows for state='{cg.STATE}'. Run: python manage.py fetch_wrldc_psp --years 2026")

        w("  Training XGBoost + Prophet (this can take a few seconds)…")
        try:
            xgb, prophet, meta = cg.train(test_frac=test_frac)
        except Exception as e:  # noqa: BLE001
            raise CommandError(f"training failed: {e}")

        cg.save_models(xgb, prophet)
        cg.write_metrics(meta)

        # ---- report -------------------------------------------------------
        w(self.style.MIGRATE_HEADING("\n=== Data ==="))
        w(f"  daily points : {meta['n_daily']:,}  "
          f"(train {meta['n_train']:,} / test {meta['n_test']:,})")
        w(f"  train range  : {meta['train_range'][0]} -> {meta['train_range'][1]}")
        w(f"  test  range  : {meta['test_range'][0]} -> {meta['test_range'][1]}")

        w(self.style.MIGRATE_HEADING("\n=== Test-set metrics (daily peak MW) ==="))
        header = f"  {'model':<10}{'MAE':>10}{'RMSE':>10}{'MAPE %':>10}"
        w(header)
        w("  " + "-" * (len(header) - 2))
        for name in ("xgboost", "prophet"):
            m = meta["models"][name]
            w(f"  {name:<10}{m['mae']:>10.2f}{m['rmse']:>10.2f}{m['mape']:>10.3f}")

        best = min(("xgboost", "prophet"),
                   key=lambda k: meta["models"][k]["rmse"])
        w(self.style.SUCCESS(f"\n  best by RMSE : {best}"))

        w(self.style.MIGRATE_HEADING("\n=== Saved ==="))
        w(self.style.SUCCESS(f"  XGBoost  -> {cg.XGB_PATH}"))
        w(self.style.SUCCESS(f"  Prophet  -> {cg.PROPHET_PATH}"))
        w(self.style.SUCCESS(f"  metrics  -> {cg.METRICS_PATH}"))
        w(self.style.SUCCESS("\nDone. Forecast API is ready at /api/cg/forecast/?days=30"))
