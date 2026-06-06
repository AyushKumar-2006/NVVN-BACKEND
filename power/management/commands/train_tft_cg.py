"""
Train the CG 5-min Temporal Fusion Transformer.

Usage:
    # quick end-to-end validation (~1 min): tiny data slice, tiny model, 1 epoch
    python manage.py train_tft_cg --smoke

    # full training run (long; expect several hours on CPU/MPS)
    python manage.py train_tft_cg --epochs 40 --accelerator auto

Saves the artifact as power/ml/models/modelsTrainData/state_5min_tft_CG.pkl
(or ..._smoke.pkl in smoke mode).
"""

import os

from django.core.management.base import BaseCommand

from power.ml.tft.train_tft import train_cg_tft
from power.ml.tft.config import SMOKE_FILENAME, MODEL_FILENAME


class Command(BaseCommand):
    help = "Train the CG 5-min Temporal Fusion Transformer (TFT) model."

    def add_arguments(self, parser):
        parser.add_argument("--state", default="CG")
        parser.add_argument("--smoke", action="store_true",
                            help="Fast end-to-end validation run.")
        parser.add_argument("--epochs", type=int, default=40)
        parser.add_argument("--accelerator", default="auto",
                            help="auto | cpu | mps | gpu")
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--max-days", type=int, default=None,
                            help="Train on only the most recent N days "
                                 "(faster, lower quality; None = all data).")

    def handle(self, *args, **opts):
        # Let unsupported MPS ops fall back to CPU instead of crashing a long run.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        smoke = opts["smoke"]
        out_filename = SMOKE_FILENAME if smoke else MODEL_FILENAME

        self.stdout.write(self.style.NOTICE(
            f"Training CG TFT (smoke={smoke}, epochs={opts['epochs']}, "
            f"accelerator={opts['accelerator']}) ..."
        ))

        path = train_cg_tft(
            state=opts["state"],
            smoke=smoke,
            max_epochs=opts["epochs"],
            accelerator=opts["accelerator"],
            batch_size=opts["batch_size"],
            max_days=opts["max_days"],
            out_filename=out_filename,
            log=lambda m: self.stdout.write(str(m)),
        )

        self.stdout.write(self.style.SUCCESS(f"Done. Saved -> {path}"))
