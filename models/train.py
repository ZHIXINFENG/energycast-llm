
import pandas as pd
import lightning.pytorch as pl
from pathlib import Path
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
MAX_ENCODER_LENGTH = 168
MAX_PREDICTION_LENGTH = 24

def load_datasets():
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val_df   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    combined = pd.concat([train_df, val_df]).reset_index(drop=False)
    training = TimeSeriesDataSet(
        combined[combined["timestamp"] <= train_df.index.max()],
        time_idx="time_idx", target="load", group_ids=["series_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=["time_idx","hour","day_of_week","month","is_weekend"],
        time_varying_unknown_reals=["load","lag_24h","lag_168h"],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        combined[combined["timestamp"] > train_df.index.max()],
        predict=True, stop_randomization=True,
    )
    return training, validation

def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print("Loading datasets...")
    training, validation = load_datasets()
    train_loader = training.to_dataloader(train=True,  batch_size=64, num_workers=2)
    val_loader   = validation.to_dataloader(train=False, batch_size=64, num_workers=2)
    model = TemporalFusionTransformer.from_dataset(
        training, learning_rate=3e-3, hidden_size=32,
        attention_head_size=2, dropout=0.1, hidden_continuous_size=16,
        loss=QuantileLoss(), log_interval=50, reduce_on_plateau_patience=4,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(dirpath=CHECKPOINT_DIR, filename="tft-best",
                        monitor="val_loss", save_top_k=1),
    ]
    trainer = pl.Trainer(max_epochs=30, accelerator="auto",
                         gradient_clip_val=0.1, callbacks=callbacks)
    print("Training...")
    trainer.fit(model, train_loader, val_loader)
    print("Done.")

if __name__ == "__main__":
    main()
