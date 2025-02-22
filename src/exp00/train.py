import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import pandas as pd
import json
from pathlib import Path

from model import Model0
from dataset import Dataset0, get_max_size
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import albumentations as A
from transformers import get_cosine_schedule_with_warmup


class CFG:
    data_dir = Path("input/data_10hz")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    time_window = 1

    max_epochs = 10
    batch_size = 32
    lr = 1e-3

    train_transform = A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    predict_videos = ['2024-11-10-18-33-49.mp4', '2024-11-10-19-21-45.mp4', '2025-01-04_08-37-18.mp4', '2025-01-04_08-40-12.mp4']

def sample_negative_frames(df: pd.DataFrame) -> pd.DataFrame:
    df_pos = df[df["label_prob"] != 0]
    df_neg = df[df["label_prob"] == 0]
    max_label_count = df_pos["action_id"].value_counts().max()
    df_neg = df_neg.sample(n=max_label_count, replace=True)
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    return df


def calculate_label_prob_for_video(df_video: pd.DataFrame, time_window: int) -> pd.DataFrame:
    """
    1) labelが NaN でなければ label_prob = 1
    2) NaN の場合、前後3フレームに NaN でない label があれば、
       (0.5)^(そのフレームまでの frame_id 差) を label_prob とする
    3) 前後3フレームに NaN でない label が一つも無い場合は label_prob = 0
    """
    # frame_id 昇順にソート（念のため）
    df_video = df_video.sort_values("frame_idx").reset_index(drop=True)
    
    action_id_list = []
    label_prob_list = []
    frame_idxs = df_video["frame_idx"].values
    action_ids = df_video["action_id"].values
    n = len(df_video)
    
    for i in range(n):
        # 1) 自身の label が NaN でなければ 1
        if action_ids[i] != 0:
            action_id_list.append(action_ids[i])
            label_prob_list.append(1.0)
            continue
        
        # 2) NaN の場合、前後3フレーム内に NaN でない label があるか探索
        start_idx = max(0, i - time_window)
        end_idx = min(n, i + time_window + 1)  # i+3 まで含めるため +4
        distances = []
        
        for j in range(start_idx, end_idx):
            # ラベルが存在するフレームを見つけたら距離を計算
            if action_ids[j] != 0:
                dist = abs(frame_idxs[i] - frame_idxs[j])
                distances.append(dist)
        
        # 距離の中で最小のものを使って (0.5)^dist
        if len(distances) == 0:
            action_id_list.append(0)
            label_prob_list.append(0.0)
        else:
            min_dist = min(distances)
            action_id_list.append(action_ids[i])
            label_prob_list.append(0.5 ** min_dist)
    
    df_video["action_id"] = action_id_list
    df_video["label_prob"] = label_prob_list
    return df_video


def preprocess_frame_label_df(df: pd.DataFrame) -> pd.DataFrame:
    df["action"] = df["labels"].str.split("_").str[1]

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

    df = df.groupby("video_filename").apply(calculate_label_prob_for_video, time_window=3)
    # df["label_prob"] = np.where(df["action_id"] != 0, 1.0, 0.0)
    return df


class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)

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
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        train_steps = self.trainer.estimated_stepping_batches
        print(train_steps)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_steps//CFG.max_epochs, num_training_steps=train_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class DataModule(L.LightningDataModule):
    def __init__(self, train_df, val_df, pred_df, max_height, max_width):
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
        self.pred_ds = Dataset0(
            frame_label_df=pred_df,
            frame_dir=CFG.data_dir / "frames",
            time_window=CFG.time_window,
            max_height=max_height,
            max_width=max_width,
            action_to_id=CFG.metadata["action_to_id"],
            num_classes=CFG.num_classes,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=16)


if __name__ == "__main__":
    frame_label_df = pd.read_csv(CFG.data_dir / "frame_label.csv")
    frame_label_df = preprocess_frame_label_df(frame_label_df)

    max_height, max_width = get_max_size(frame_label_df, CFG.data_dir / "frames")

    pred_df = frame_label_df[frame_label_df["video_filename"].isin(CFG.predict_videos)]
    frame_label_df = frame_label_df[~frame_label_df["video_filename"].isin(CFG.predict_videos)]

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
    valid_df = frame_label_df.iloc[val_indices]
    valid_df = sample_negative_frames(valid_df)

    data_module = DataModule(train_df, valid_df, pred_df, max_height, max_width)

    model = Model0(num_classes=CFG.num_classes, backbone="resnet34d")
    lit_model = LitModel(model, lr=CFG.lr)

    callbacks = [
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, save_last=True),
    ]
    trainer = L.Trainer(max_epochs=CFG.max_epochs, callbacks=callbacks)
    trainer.fit(lit_model, data_module)

    # lit_model.load_state_dict(torch.load("lightning_logs/version_13/checkpoints/epoch=5-step=498.ckpt")["state_dict"])    
    preds = trainer.predict(lit_model, data_module, return_predictions=True)
    preds = torch.cat(preds, dim=0)
    
    id_to_action = {v: k for k, v in CFG.metadata["action_to_id"].items()}
    pred_df["pred_action_id"] = preds.argmax(dim=1)
    pred_df["pred_action"] = pred_df["pred_action_id"].map(id_to_action)
    pred_df.to_csv("pred_result.csv", index=False)
