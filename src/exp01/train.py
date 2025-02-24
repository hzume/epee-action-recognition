from sklearn.metrics import f1_score
import torch
from pprint import pprint
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import pandas as pd
import json
from pathlib import Path

from model import Model1
from dataset import Dataset1, get_max_size
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import albumentations as A
from transformers import get_cosine_schedule_with_warmup


class CFG:
    data_dir = Path("input/data_10hz")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    time_window = 3
    width = 3

    backbone = "resnet34d"
    # backbone = "vit_small_patch16_224.augreg_in21k_ft_in1k"

    max_epochs = 10
    batch_size = 32
    lr = 1e-3

    train_transform = A.ReplayCompose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.ISONoise(p=0.3),
            # A.MotionBlur(p=0.3),
            # A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-10, 10), p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ]
    )
    valid_transform = A.ReplayCompose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    predict_videos = ['2024-11-10-18-33-49.mp4', '2024-11-10-19-21-45.mp4', '2025-01-04_08-37-18.mp4', '2025-01-04_08-40-12.mp4']


class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        x = x.float()
        y_left = y_left.float()
        y_right = y_right.float()
        y_hat_left, y_hat_right = self(x)
        loss_left = F.cross_entropy(y_hat_left, y_left)
        loss_right = F.cross_entropy(y_hat_right, y_right)
        loss = (loss_left + loss_right) / 2
        accuracy_left = (y_hat_left.argmax(dim=1) == y_left.argmax(dim=1)).float().mean()
        accuracy_right = (y_hat_right.argmax(dim=1) == y_right.argmax(dim=1)).float().mean()
        accuracy = (accuracy_left + accuracy_right) / 2
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "train_accuracy",
            accuracy,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        x = x.float()
        y_left = y_left.float()
        y_right = y_right.float()
        y_hat_left, y_hat_right = self(x)
        loss_left = F.cross_entropy(y_hat_left, y_left)
        loss_right = F.cross_entropy(y_hat_right, y_right)
        loss = (loss_left + loss_right) / 2
        accuracy_left = (y_hat_left.argmax(dim=1) == y_left.argmax(dim=1)).float().mean()
        accuracy_right = (y_hat_right.argmax(dim=1) == y_right.argmax(dim=1)).float().mean()
        accuracy = (accuracy_left + accuracy_right) / 2
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc", accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        x = x.float()
        y_left = y_left.float()
        y_right = y_right.float()
        y_hat_left, y_hat_right = self(x)
        y_hat_left = torch.softmax(y_hat_left, dim=1)
        y_hat_right = torch.softmax(y_hat_right, dim=1)
        return y_hat_left, y_hat_right

    def configure_optimizers(self):
        train_steps = self.trainer.estimated_stepping_batches
        print(train_steps)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_steps//CFG.max_epochs, num_training_steps=train_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

def prepare_label_df(df: pd.DataFrame, action_to_id: dict[str, int]) -> pd.DataFrame:
    def parse_actions(actions_str: str) -> list[int]:
            if pd.isna(actions_str):
                return [0]
            actions = actions_str.split(",")
            return list(set([action_to_id[action] for action in actions]))
            # return [action_to_id[actions[0]]]

    df["left_action_ids"] = df["left_actions"].apply(parse_actions)
    df["right_action_ids"] = df["right_actions"].apply(parse_actions)

    assert (df["left_action_ids"].apply(lambda x: (0 in x) and (len(x) > 1)).sum() == 0)
    assert (df["right_action_ids"].apply(lambda x: (0 in x) and (len(x) > 1)).sum() == 0)
    return df


class DataModule(L.LightningDataModule):
    def __init__(self, train_df, valid_df, pred_df, max_height, max_width):
        super().__init__()
        self.train_ds = Dataset1(
            frame_label_df=train_df,
            frame_dir=CFG.data_dir / "frames",
            time_window=CFG.time_window,
            width=CFG.width,
            max_height=max_height,
            max_width=max_width,
            action_to_id=CFG.metadata["action_to_id"],
            num_classes=CFG.num_classes,
            transforms=CFG.train_transform,
        )
        self.val_ds = Dataset1(
            frame_label_df=valid_df,
            frame_dir=CFG.data_dir / "frames",
            time_window=CFG.time_window,
            width=CFG.width,
            max_height=max_height,
            max_width=max_width,
            action_to_id=CFG.metadata["action_to_id"],
            num_classes=CFG.num_classes,
            transforms=CFG.valid_transform,
            sampling=True,
            p_flip=0.0,
        )
        self.pred_ds = Dataset1(
            frame_label_df=pred_df,
            frame_dir=CFG.data_dir / "frames",
            time_window=CFG.time_window,
            width=CFG.width,
            max_height=max_height,
            max_width=max_width,
            action_to_id=CFG.metadata["action_to_id"],
            num_classes=CFG.num_classes,
            sampling=False,
            p_flip=0.0,
            transforms=CFG.valid_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=16)


def train_valid_split(frame_label_df: pd.DataFrame) -> tuple[list[int], list[int]]:
    video_labels = frame_label_df[["video_filename", "left_action_ids", "right_action_ids"]].copy()
    action_ids = []
    for _, row in video_labels.iterrows():
        new_list = []
        new_list.extend(row["left_action_ids"])
        new_list.extend(row["right_action_ids"])
        new_list = list(set(new_list))
        action_ids.append(new_list)
    video_labels["action_ids"] = action_ids
    video_labels = video_labels.drop(columns=["left_action_ids", "right_action_ids"])
    video_labels = video_labels.explode("action_ids")
    video_labels["action_ids"] = video_labels["action_ids"].astype(int)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    train_indices, val_indices = next(
        skf.split(
            video_labels,
            video_labels["action_ids"],
            video_labels["video_filename"],
        )
    )
    train_videos = video_labels.iloc[train_indices]["video_filename"].unique()
    val_videos = video_labels.iloc[val_indices]["video_filename"].unique()
    return train_videos, val_videos


if __name__ == "__main__":
    frame_label_df = pd.read_csv(CFG.data_dir / "frame_label.csv")
    frame_label_df = prepare_label_df(frame_label_df, CFG.metadata["action_to_id"])
    max_height, max_width = get_max_size(frame_label_df, CFG.data_dir / "frames")

    pred_df = frame_label_df[frame_label_df["video_filename"].isin(CFG.predict_videos)]
    frame_label_df = frame_label_df[~frame_label_df["video_filename"].isin(CFG.predict_videos)]

    train_videos, val_videos = train_valid_split(frame_label_df)
    pprint(val_videos.tolist())
    train_df = frame_label_df[frame_label_df["video_filename"].isin(train_videos)]
    valid_df = frame_label_df[frame_label_df["video_filename"].isin(val_videos)]

    data_module = DataModule(train_df, valid_df, pred_df, max_height, max_width)

    model = Model1(num_classes=CFG.num_classes, backbone=CFG.backbone)
    lit_model = LitModel(model, lr=CFG.lr)

    callbacks = [
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, save_last=True),
    ]
    trainer = L.Trainer(max_epochs=CFG.max_epochs, callbacks=callbacks)
    trainer.fit(lit_model, data_module)



    # ################################################################################################################
    # valid_predictions = trainer.predict(lit_model, data_module.val_dataloader(), return_predictions=True)
    # y_hat_left_list, y_hat_right_list = zip(*valid_predictions)
    # y_hat_left = torch.cat(y_hat_left_list, dim=0)
    # y_hat_right = torch.cat(y_hat_right_list, dim=0)

    # print(data_module.val_ds.frame_label_df["video_filename"].unique())

    # y_left_arr, y_right_arr = [], []
    # for x, y_left, y_right in data_module.val_dataloader():
    #     y_left_arr.append(y_left)
    #     y_right_arr.append(y_right)
    # y_left = torch.cat(y_left_arr, dim=0)
    # y_right = torch.cat(y_right_arr, dim=0)

    # acc_none_left = ((y_hat_left.argmax(dim=1)==0) == (y_left.argmax(dim=1)==0)).float().mean()
    # acc_none_right = ((y_hat_right.argmax(dim=1)==0) == (y_right.argmax(dim=1)==0)).float().mean()
    # acc_none = (acc_none_left + acc_none_right) / 2
    # print(f"val_acc_none: {acc_none}")

    # y_hat_left_pos = y_hat_left.argmax(dim=1)[y_left.argmax(dim=1) != 0]
    # y_hat_right_pos = y_hat_right.argmax(dim=1)[y_right.argmax(dim=1) != 0]
    # y_left_pos = y_left.argmax(dim=1)[y_left.argmax(dim=1) != 0]
    # y_right_pos = y_right.argmax(dim=1)[y_right.argmax(dim=1) != 0]
    # acc_ignore_none_left = (y_hat_left_pos == y_left_pos).float().mean()
    # acc_ignore_none_right = (y_hat_right_pos == y_right_pos).float().mean()
    # acc_ignore_none = (acc_ignore_none_left + acc_ignore_none_right) / 2
    # print(f"val_acc_ignore_none: {acc_ignore_none}")
    # ################################################################################################################



    # lit_model.load_state_dict(torch.load("lightning_logs/version_11/checkpoints/last.ckpt")["state_dict"])    
    predictions = trainer.predict(lit_model, data_module.predict_dataloader(), return_predictions=True)
    y_hat_left_list, y_hat_right_list = zip(*predictions)
    y_hat_left = torch.cat(y_hat_left_list, dim=0)
    y_hat_right = torch.cat(y_hat_right_list, dim=0)
    
    id_to_action = {v: k for k, v in CFG.metadata["action_to_id"].items()}
    pred_df = data_module.pred_ds.frame_label_df[["video_filename", "frame_idx", "left_actions", "right_actions"]].copy()
    for i in range(CFG.num_classes):
        if i == 0:
            action_name = "none"
        else:
            action_name = id_to_action[i]
        pred_df[f"left_pred_action_{action_name}"] = y_hat_left[:, i]
        pred_df[f"right_pred_action_{action_name}"] = y_hat_right[:, i]
    pred_df["left_pred_action"] = y_hat_left.argmax(dim=1)
    pred_df["right_pred_action"] = y_hat_right.argmax(dim=1)
    pred_df["left_pred_action"] = pred_df["left_pred_action"].map(id_to_action)
    pred_df["right_pred_action"] = pred_df["right_pred_action"].map(id_to_action)
    pred_df.to_csv("pred_result.csv", index=False)
