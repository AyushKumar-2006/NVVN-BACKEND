"""
Train the CG 5-min Temporal Fusion Transformer.

The whole trained artifact (best Lightning checkpoint + the dataset parameters
needed to rebuild the model + the seasonal load profile + weather climatology)
is packaged into a single joblib file: state_5min_tft_CG.pkl — saved next to the
existing XGBoost models via the shared model_store.
"""

import io
import os
import tempfile
from datetime import datetime

import joblib
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from power.ml.model_store import PATH as MODEL_PATH
from power.ml.tft.config import (
    ARTIFACT_FORMAT,
    GROUP_COL,
    KNOWN_REALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    MODEL_FILENAME,
    TARGET,
    TIME_IDX,
    UNKNOWN_REALS,
)
from power.ml.tft.features_tft import build_training_frame


def _build_datasets(df, encoder_length, prediction_length, val_days):
    """Training dataset on all-but-last `val_days`, validation on the tail."""
    val_steps = val_days * prediction_length
    training_cutoff = int(df[TIME_IDX].max()) - val_steps

    training = TimeSeriesDataSet(
        df[df[TIME_IDX] <= training_cutoff],
        time_idx=TIME_IDX,
        target=TARGET,
        group_ids=[GROUP_COL],
        max_encoder_length=encoder_length,
        max_prediction_length=prediction_length,
        static_categoricals=[GROUP_COL],
        time_varying_known_reals=[TIME_IDX] + KNOWN_REALS,
        time_varying_unknown_reals=list(UNKNOWN_REALS),
        target_normalizer=GroupNormalizer(groups=[GROUP_COL], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=training_cutoff + 1,
        stop_randomization=True,
    )
    return training, validation


def train_cg_tft(
    state: str = "CG",
    *,
    smoke: bool = False,
    max_epochs: int = 40,
    accelerator: str = "auto",
    batch_size: int = 128,
    hidden_size: int = 32,
    attention_head_size: int = 2,
    hidden_continuous_size: int = 16,
    learning_rate: float = 0.03,
    dropout: float = 0.1,
    patience: int = 6,
    max_days: int | None = None,
    out_filename: str | None = None,
    log=print,
):
    """Train and persist the TFT. Returns the absolute path of the saved pkl."""

    # Smoke mode: tiny data slice + tiny model + 1 epoch -> validates the full
    # train -> save -> load -> predict cycle in well under a minute.
    if smoke:
        max_epochs = 1
        encoder_length = MAX_PREDICTION_LENGTH        # 1 day of context
        prediction_length = MAX_PREDICTION_LENGTH
        val_days = 1
        max_days = 30
        hidden_size, attention_head_size, hidden_continuous_size = 8, 1, 4
    else:
        encoder_length = MAX_ENCODER_LENGTH
        prediction_length = MAX_PREDICTION_LENGTH
        # On a reduced-data run keep validation short so most of the slice trains.
        val_days = 7 if max_days is not None else 14
        # max_days flows through from the caller (None = use all available data).

    log(f"[TFT] building training frame for {state} (max_days={max_days}) ...")
    df, profile, weather_clim = build_training_frame(state, max_days=max_days)
    log(f"[TFT] frame ready: rows={len(df)} cols={df.shape[1]} "
        f"range={df['ds'].min()} -> {df['ds'].max()}")

    training, validation = _build_datasets(df, encoder_length, prediction_length, val_days)

    train_dl = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(),
        optimizer="adam",
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )
    log(f"[TFT] model params: {tft.size()/1e3:.1f}k")

    ckpt_dir = tempfile.mkdtemp(prefix="tft_ckpt_")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val_loss", mode="min", save_top_k=1,
    )
    callbacks = [checkpoint_cb]
    if not smoke:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=patience, mode="min"))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=8 if smoke else 1.0,
        limit_val_batches=4 if smoke else 1.0,
    )

    log(f"[TFT] training start (max_epochs={max_epochs}, accelerator={accelerator}) ...")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    log("[TFT] training done.")

    best_path = checkpoint_cb.best_model_path or os.path.join(ckpt_dir, "best.ckpt")
    if not best_path or not os.path.exists(best_path):
        # No checkpoint was written (e.g. smoke with 0 val improvement) -> dump current.
        best_path = os.path.join(ckpt_dir, "manual.ckpt")
        trainer.save_checkpoint(best_path)
    log(f"[TFT] best checkpoint: {best_path}")

    with open(best_path, "rb") as fh:
        checkpoint_bytes = fh.read()

    artifact = {
        "format": ARTIFACT_FORMAT,
        "state": state,
        "trained_at": datetime.utcnow().isoformat(),
        "smoke": smoke,
        "checkpoint_bytes": checkpoint_bytes,
        "dataset_parameters": training.get_parameters(),
        "profile": profile,                      # mm-dd, slot -> profile_y
        "weather_climatology": weather_clim,     # mm-dd, hour -> mean weather
        "known_reals": list(KNOWN_REALS),
        "max_encoder_length": encoder_length,
        "max_prediction_length": prediction_length,
        "last_datetime": df["ds"].max().isoformat(),
        "data_rows": int(len(df)),
    }

    out_filename = out_filename or MODEL_FILENAME
    os.makedirs(MODEL_PATH, exist_ok=True)
    out_path = os.path.join(MODEL_PATH, out_filename)
    joblib.dump(artifact, out_path)
    log(f"[TFT] saved artifact -> {out_path}  ({len(checkpoint_bytes)/1e6:.1f} MB ckpt)")

    return out_path


def load_tft_artifact(filename: str = MODEL_FILENAME):
    """Load the packaged artifact and rebuild the trained TFT model (on CPU)."""
    path = os.path.join(MODEL_PATH, filename)
    artifact = joblib.load(path)

    with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as fh:
        fh.write(artifact["checkpoint_bytes"])
        ckpt_path = fh.name
    try:
        model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path, map_location="cpu")
    finally:
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    model.eval()
    return model, artifact
