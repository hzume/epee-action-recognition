import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader
import pandas as pd
import json
from pathlib import Path

from model import Model0
from dataset import Dataset0, get_max_size, sample_negative_frames
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import albumentations as A


class CFG:
    data_dir = Path("input/data")
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    time_window = 2

    batch_size = 2

    train_transform = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def preprocess_frame_label_df(df: pd.DataFrame) -> pd.DataFrame:
    df["action_id"] = np.where(
        df["action"].isna(),
        0,
        df["action"].map(CFG.metadata["action_to_id"]),
    ).astype(int)

    frame_min_idx = (
        df.groupby("video_filename")["frame_idx"]
        .min()
        .reset_index(name="frame_idx_min")
    )
    frame_max_idx = (
        df.groupby("video_filename")["frame_idx"]
        .max()
        .reset_index(name="frame_idx_max")
    )
    df = df.merge(frame_min_idx, on="video_filename")
    df = df.merge(frame_max_idx, on="video_filename")

    is_in_time_window = df["frame_idx"].between(
        df["frame_idx_min"] + CFG.time_window,
        df["frame_idx_max"] - CFG.time_window,
    )
    df = df[is_in_time_window]
    frame_paths = []
    for _, row in df.iterrows():
        frame_idx = row["frame_idx"]
        video_basename = row["video_filename"].replace(".mp4", "")
        frame_idxs = list(
            range(frame_idx - CFG.time_window, frame_idx + CFG.time_window + 1)
        )
        frame_paths.append(
            [
                str(CFG.data_dir / "frames" / f"{video_basename}_{frame_idx}.jpg")
                for frame_idx in frame_idxs
            ]
        )
    df["frame_paths"] = frame_paths
    return df


class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class DataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, max_height, max_width):
        super().__init__()
        self.train_ds = Dataset0(
            frame_label_df=train_df,
            frame_dir=CFG.data_dir / "frames",
            time_window=CFG.time_window,
            max_height=max_height,
            max_width=max_width,
            action_to_id=CFG.metadata["action_to_id"],
            num_classes=CFG.num_classes,
        )
        self.val_ds = Dataset0(
            frame_label_df=val_df,
            frame_dir=CFG.data_dir / "frames",
            time_window=CFG.time_window,
            max_height=max_height,
            max_width=max_width,
            action_to_id=CFG.metadata["action_to_id"],
            num_classes=CFG.num_classes,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=CFG.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=CFG.batch_size, shuffle=False)


if __name__ == "__main__":
    frame_label_df = pd.read_csv(CFG.data_dir / "frame_label.csv")
    frame_label_df = preprocess_frame_label_df(frame_label_df)

    max_height, max_width = get_max_size(frame_label_df, CFG.data_dir / "frames")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, val_indices = next(
        skf.split(
            frame_label_df,
            frame_label_df["action_id"],
            frame_label_df["video_filename"],
        )
    )
    train_df = frame_label_df.iloc[train_indices]
    train_df = sample_negative_frames(train_df)
    val_df = frame_label_df.iloc[val_indices]
    val_df = sample_negative_frames(val_df)

    data_module = DataModule(train_df, val_df, max_height, max_width)

    model = Model0(num_classes=CFG.num_classes, backbone="resnet34d")
    lit_model = LitModel(model)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(lit_model, data_module)
