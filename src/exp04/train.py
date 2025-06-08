from pprint import pprint
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import functional as F
import torch
import pytorch_lightning as L
import warnings
from transformers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings("ignore")

class CFG:
    data_dir = Path("input/data_10hz")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    batch_size = 32
    epochs = 50

    num_kp = 17
    window_width = 3
    predict_videos = ['2024-11-10-18-33-49.mp4', '2024-11-10-19-21-45.mp4', '2025-01-04_08-37-18.mp4', '2025-01-04_08-40-12.mp4']

class EpeeDataset(Dataset):
    def __init__(self, raw_df: pd.DataFrame, sample: bool = False):
        self.action_to_id = CFG.metadata["action_to_id"]
        self.df = self.preprocess(raw_df)
        self.feature_cols = list(set(self.df.columns.tolist()) - {"frame_filename", "video_filename", "frame_idx", "left_action_id", "right_action_id", "switched"})
        data = []
        for switched in [False, True]:
            for video_filename in tqdm(self.df["video_filename"].unique().tolist()):
                df_group: pd.DataFrame = self.df[(self.df["video_filename"] == video_filename) & (self.df["switched"] == switched)].sort_values(by="frame_idx")
                frame_idxs = set(df_group["frame_idx"].values)
                last_frame_idx = max(frame_idxs)

                for i in range(last_frame_idx):
                    window = set(range(i - CFG.window_width, i + CFG.window_width + 1))
                    if window.issubset(frame_idxs):
                        x = torch.tensor(df_group[df_group["frame_idx"].isin(window)][self.feature_cols].values).float()
                        y_left = F.one_hot(torch.tensor(df_group[df_group["frame_idx"] == i]["left_action_id"].values).squeeze(), num_classes=CFG.num_classes).float()
                        y_right = F.one_hot(torch.tensor(df_group[df_group["frame_idx"] == i]["right_action_id"].values).squeeze(), num_classes=CFG.num_classes).float()
                        frame_filename = df_group[df_group["frame_idx"] == i]["frame_filename"].values[0]
                        data.append({
                            "video_filename": video_filename,
                            "frame_filename": frame_filename,
                            "frame_idx": i,
                            "switched": switched,
                            "x": x,
                            "y_left": y_left,
                            "y_right": y_right,
                        })
        
        if sample:
            data_positive = [data[i] for i in range(len(data)) if data[i]["y_left"].argmax().item() != 0 or data[i]["y_right"].argmax().item() != 0]
            data_negative = [data[i] for i in range(len(data)) if data[i]["y_left"].argmax().item() == 0 and data[i]["y_right"].argmax().item() == 0]
            data_negative = random.sample(data_negative, len(data_positive))
            data = data_positive + data_negative

        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]["x"], self.data[idx]["y_left"], self.data[idx]["y_right"]

    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = self.process_raw_df(raw_df)
        df_switched = self.switch_side(df)
        
        df["switched"] = False
        df_switched["switched"] = True

        df = pd.concat([df, df_switched])
        df = self.add_pose_features(df)

        return df

    def process_raw_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        raw_df["bbox_area"] = (raw_df["bbox_x_2"] - raw_df["bbox_x_1"]) * (raw_df["bbox_y_2"] - raw_df["bbox_y_1"])
        raw_df["bbox_ratio"] = raw_df["bbox_area"] / (raw_df["width"] * raw_df["height"])
        raw_df["min_keypoint_score"] = raw_df[[f"keypoint_score_{i}" for i in range(CFG.num_kp)]].min(axis=1)
        raw_df["center_x"] = (raw_df["bbox_x_1"] + raw_df["bbox_x_2"]) / 2
        raw_df["center_y"] = (raw_df["bbox_y_1"] + raw_df["bbox_y_2"]) / 2

        # filter trying to extract target player bboxes
        raw_df = raw_df[(raw_df["min_keypoint_score"] > 0.2) & (raw_df["bbox_ratio"] > 0.0375) & (raw_df["bbox_ratio"] < 0.15)]

        data = {
            "frame_filename": [],
            "video_filename": [],
            "frame_idx": [],
            "height": [],
            "width": [],
            "left_action_id": [],
            "right_action_id": [],
        } | {f"left_keypoint_{i}_x": [] for i in range(CFG.num_kp)} \
        | {f"left_keypoint_{i}_y": [] for i in range(CFG.num_kp)} \
        | {f"right_keypoint_{i}_x": [] for i in range(CFG.num_kp)} \
        | {f"right_keypoint_{i}_y": [] for i in range(CFG.num_kp)} \
        | {"left_center_x": [], "left_center_y": [], "right_center_x": [], "right_center_y": []} \
        | {"left_bbox_area": [], "right_bbox_area": []}
        
        for frame_filename, df_frame in raw_df.groupby("frame_filename"):
            if len(df_frame) == 1:
                continue

            target_rows = df_frame.sort_values(by="min_keypoint_score", ascending=False)[:2]

            # 片方の中心のy座標がもう片方のbboxの外側にある場合は除外
            if target_rows["center_y"].iloc[0] > target_rows["bbox_y_2"].iloc[1] \
                or target_rows["center_y"].iloc[0] < target_rows["bbox_y_1"].iloc[1] \
                or target_rows["center_y"].iloc[1] > target_rows["bbox_y_2"].iloc[0] \
                or target_rows["center_y"].iloc[1] < target_rows["bbox_y_1"].iloc[0]:
                continue

            data["frame_filename"].append(frame_filename)
            data["video_filename"].append(df_frame["video_filename"].iloc[0])
            data["frame_idx"].append(df_frame["frame_idx"].iloc[0])
            data["height"].append(df_frame["height"].iloc[0])
            data["width"].append(df_frame["width"].iloc[0])
            data["left_action_id"].append(self.action_to_id[df_frame["left_action"].iloc[0]] if (df_frame["left_action"].iloc[0] in self.action_to_id) else 0)
            data["right_action_id"].append(self.action_to_id[df_frame["right_action"].iloc[0]] if (df_frame["right_action"].iloc[0] in self.action_to_id) else 0)
            
            if target_rows["center_x"].iloc[0] < target_rows["center_x"].iloc[1]:
                left_row = target_rows.iloc[0]
                right_row = target_rows.iloc[1]
            else:
                left_row = target_rows.iloc[1]
                right_row = target_rows.iloc[0]
                
            for i in range(CFG.num_kp):
                data[f"left_keypoint_{i}_x"].append(left_row[f"keypoint_{i}_x"])
                data[f"left_keypoint_{i}_y"].append(left_row[f"keypoint_{i}_y"])
                data[f"right_keypoint_{i}_x"].append(right_row[f"keypoint_{i}_x"])
                data[f"right_keypoint_{i}_y"].append(right_row[f"keypoint_{i}_y"])
                
            data["left_center_x"].append(left_row["center_x"])
            data["left_center_y"].append(left_row["center_y"])
            data["left_bbox_area"].append(left_row["bbox_area"])    
            
            data["right_center_x"].append(right_row["center_x"])
            data["right_center_y"].append(right_row["center_y"])
            data["right_bbox_area"].append(right_row["bbox_area"])

        df = pd.DataFrame(data)
        
        for i in range(CFG.num_kp):
            df[f"left_keypoint_{i}_x"] = (df[f"left_keypoint_{i}_x"] - df["left_center_x"]) / df["left_bbox_area"]
            df[f"left_keypoint_{i}_y"] = (df[f"left_keypoint_{i}_y"] - df["left_center_y"]) / df["left_bbox_area"]

            df[f"right_keypoint_{i}_x"] = (df[f"right_keypoint_{i}_x"] - df["right_center_x"]) / df["right_bbox_area"]
            df[f"right_keypoint_{i}_y"] = (df[f"right_keypoint_{i}_y"] - df["right_center_y"]) / df["right_bbox_area"]
        
        df["distance"] = (df["right_center_x"] - df["left_center_x"]) / (df["right_bbox_area"] + df["left_bbox_area"])
        
        df = df.drop(columns=["left_center_x", "left_center_y", "right_center_x", "right_center_y", "left_bbox_area", "right_bbox_area", "height", "width"])
        return df

    def add_pose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for i in range(CFG.num_kp):
            df[f"right_keypoint_{i}_dist"] = np.sqrt(df[f"right_keypoint_{i}_x"] ** 2 + df[f"right_keypoint_{i}_y"] ** 2)
            df[f"right_keypoint_{i}_angle"] = np.arctan2(df[f"right_keypoint_{i}_y"], df[f"right_keypoint_{i}_x"])

            df[f"left_keypoint_{i}_dist"] = np.sqrt(df[f"left_keypoint_{i}_x"] ** 2 + df[f"left_keypoint_{i}_y"] ** 2)
            df[f"left_keypoint_{i}_angle"] = np.arctan2(df[f"left_keypoint_{i}_y"], df[f"left_keypoint_{i}_x"])

        df["left_l_shoulder_elbow_angle"] = np.arctan2((df["left_keypoint_7_y"] - df["left_keypoint_5_y"]), (df["left_keypoint_7_x"] - df["left_keypoint_5_x"]))
        df["left_r_shoulder_elbow_angle"] = np.arctan2((df["left_keypoint_8_y"] - df["left_keypoint_6_y"]), (df["left_keypoint_8_x"] - df["left_keypoint_6_x"]))
        df["left_l_elbow_wrist_angle"] = np.arctan2((df["left_keypoint_9_y"] - df["left_keypoint_7_y"]), (df["left_keypoint_9_x"] - df["left_keypoint_7_x"]))
        df["left_r_elbow_wrist_angle"] = np.arctan2((df["left_keypoint_10_y"] - df["left_keypoint_8_y"]), (df["left_keypoint_10_x"] - df["left_keypoint_8_x"]))
        df["left_l_shoulder_hip_angle"] = np.arctan2((df["left_keypoint_11_y"] - df["left_keypoint_5_y"]), (df["left_keypoint_11_x"] - df["left_keypoint_5_x"]))
        df["left_r_shoulder_hip_angle"] = np.arctan2((df["left_keypoint_12_y"] - df["left_keypoint_6_y"]), (df["left_keypoint_12_x"] - df["left_keypoint_6_x"]))
        df["left_l_hip_knee_angle"] = np.arctan2((df["left_keypoint_13_y"] - df["left_keypoint_11_y"]), (df["left_keypoint_13_x"] - df["left_keypoint_11_x"]))
        df["left_r_hip_knee_angle"] = np.arctan2((df["left_keypoint_14_y"] - df["left_keypoint_12_y"]), (df["left_keypoint_14_x"] - df["left_keypoint_12_x"]))
        df["left_l_knee_ankle_angle"] = np.arctan2((df["left_keypoint_15_y"] - df["left_keypoint_13_y"]), (df["left_keypoint_15_x"] - df["left_keypoint_13_x"]))
        df["left_r_knee_ankle_angle"] = np.arctan2((df["left_keypoint_16_y"] - df["left_keypoint_14_y"]), (df["left_keypoint_16_x"] - df["left_keypoint_14_x"]))

        df["right_l_shoulder_elbow_angle"] = np.arctan2((df["right_keypoint_7_y"] - df["right_keypoint_5_y"]), (df["right_keypoint_7_x"] - df["right_keypoint_5_x"]))
        df["right_r_shoulder_elbow_angle"] = np.arctan2((df["right_keypoint_8_y"] - df["right_keypoint_6_y"]), (df["right_keypoint_8_x"] - df["right_keypoint_6_x"]))
        df["right_l_elbow_wrist_angle"] = np.arctan2((df["right_keypoint_9_y"] - df["right_keypoint_7_y"]), (df["right_keypoint_9_x"] - df["right_keypoint_7_x"]))
        df["right_r_elbow_wrist_angle"] = np.arctan2((df["right_keypoint_10_y"] - df["right_keypoint_8_y"]), (df["right_keypoint_10_x"] - df["right_keypoint_8_x"]))

        df["right_l_hip_knee_angle"] = np.arctan2((df["right_keypoint_13_y"] - df["right_keypoint_11_y"]), (df["right_keypoint_13_x"] - df["right_keypoint_11_x"]))
        df["right_r_hip_knee_angle"] = np.arctan2((df["right_keypoint_14_y"] - df["right_keypoint_12_y"]), (df["right_keypoint_14_x"] - df["right_keypoint_12_x"]))
        df["right_l_knee_ankle_angle"] = np.arctan2((df["right_keypoint_15_y"] - df["right_keypoint_13_y"]), (df["right_keypoint_15_x"] - df["right_keypoint_13_x"]))
        df["right_r_knee_ankle_angle"] = np.arctan2((df["right_keypoint_16_y"] - df["right_keypoint_14_y"]), (df["right_keypoint_16_x"] - df["right_keypoint_14_x"]))

        return df

    def switch_side(self, df: pd.DataFrame) -> pd.DataFrame:
        df_switched = df.copy()

        kpt_map = {
            0: 0,
            1: 2,
            2: 1,
            3: 4,
            4: 3,
            5: 6,
            6: 5,
            7: 8,
            8: 7,
            9: 10,
            10: 9,
            11: 12,
            12: 11,
            13: 14,
            14: 13,
            15: 16,
            16: 15,
        }

        df_switched["left_action_id"] = df["right_action_id"]
        df_switched["right_action_id"] = df["left_action_id"]

        for i in range(CFG.num_kp):
            j = kpt_map[i]
            df_switched[f"left_keypoint_{i}_x"] = -df[f"right_keypoint_{j}_x"]
            df_switched[f"left_keypoint_{i}_y"] = df[f"right_keypoint_{j}_y"]

            df_switched[f"right_keypoint_{i}_x"] = -df[f"left_keypoint_{j}_x"]
            df_switched[f"right_keypoint_{i}_y"] = df[f"left_keypoint_{j}_y"]

        return df_switched


class EpeeDataModule(L.LightningDataModule):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        pred_df = df[df["video_filename"].isin(CFG.predict_videos)]
        df = df[~df["video_filename"].isin(CFG.predict_videos)]

        skf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        train_idx, valid_idx = next(skf.split(df, df["left_action"].fillna("none"), groups=df["video_filename"]))
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        self.train_ds = EpeeDataset(train_df, sample=True)
        self.valid_ds = EpeeDataset(valid_df)
        self.pred_ds = EpeeDataset(pred_df)

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=CFG.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=CFG.batch_size, shuffle=True)

    def predict_dataloader(self):
        return DataLoader(self.pred_ds, batch_size=CFG.batch_size, shuffle=False)



class EpeeModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=num_features, hidden_size=128, batch_first=True)
        self.fc_left = nn.Linear(128, CFG.num_classes)
        self.fc_right = nn.Linear(128, CFG.num_classes)

    def forward(self, x):
        x, (h, c) = self.rnn(x)
        last_h = h[-1]
        y_left_hat = self.fc_left(last_h)
        y_right_hat = self.fc_right(last_h)
        return y_left_hat, y_right_hat


class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float, total_steps_per_epoch: int):
        super().__init__()
        self.model = model
        self.lr = lr
        self.total_steps_per_epoch = total_steps_per_epoch

    def training_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        y_left_hat, y_right_hat = self.model(x)
        loss_left = F.cross_entropy(y_left_hat, y_left)
        loss_right = F.cross_entropy(y_right_hat, y_right)
        loss = (loss_left + loss_right) / 2
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        y_left_hat, y_right_hat = self.model(x)
        loss_left = F.cross_entropy(y_left_hat, y_left)
        loss_right = F.cross_entropy(y_right_hat, y_right)
        loss = (loss_left + loss_right) / 2

        acc_left = (y_left_hat.argmax(dim=1) == y_left.argmax(dim=1)).float().mean()
        acc_right = (y_right_hat.argmax(dim=1) == y_right.argmax(dim=1)).float().mean()
        acc = (acc_left + acc_right) / 2

        self.log("valid_loss", loss, prog_bar=True, on_epoch=True)
        self.log("valid_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        y_left_hat, y_right_hat = self.model(x)
        return y_left_hat, y_right_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.total_steps_per_epoch, num_training_steps=self.total_steps_per_epoch * CFG.epochs)
        return [optimizer], [scheduler]
        

if __name__ == "__main__":
    df = pd.read_csv(CFG.data_dir / "pose_preds.csv")
    dm = EpeeDataModule(df)

    model = EpeeModel(num_features=len(dm.train_ds.feature_cols))
    lit_model = LitModel(model, lr=1e-2, total_steps_per_epoch=len(dm.train_dataloader()))
    callbacks = [
        ModelCheckpoint(
            monitor="valid_acc",
            mode="max",
            save_top_k=1,
        )
    ]

    trainer = L.Trainer(max_epochs=CFG.epochs, callbacks=callbacks)
    trainer.fit(lit_model, dm)

    best_model_path = trainer.checkpoint_callback.best_model_path
    lit_model = LitModel.load_from_checkpoint(best_model_path, model=model, lr=1e-2, total_steps_per_epoch=len(dm.train_dataloader()))
    
    logits_left = []
    logits_right = []
    for x, y_left, y_right in dm.val_dataloader():
        y_left_hat, y_right_hat = lit_model.model(x.to(lit_model.device))
        logits_left.append(y_left_hat.detach().cpu().numpy())
        logits_right.append(y_right_hat.detach().cpu().numpy())

    logits_left = np.concatenate(logits_left, axis=0)
    logits_right = np.concatenate(logits_right, axis=0)
    pred_ids_left = np.argmax(logits_left, axis=1)
    pred_ids_right = np.argmax(logits_right, axis=1)

    true_ids_left = [dm.valid_ds.data[i]["y_left"].argmax().item() for i in range(len(dm.valid_ds))]
    true_ids_right = [dm.valid_ds.data[i]["y_right"].argmax().item() for i in range(len(dm.valid_ds))]

    result_df = pd.DataFrame({
        "left_pred_id": pred_ids_left,
        "right_pred_id": pred_ids_right,
        "left_true_id": true_ids_left,
        "right_true_id": true_ids_right,
    })

    result_df["is_none_left"] = (result_df["left_true_id"] == 0)
    result_df["is_none_right"] = (result_df["right_true_id"] == 0)

    acc_ignore_none_left = (result_df[~result_df["is_none_left"]]["left_pred_id"] == result_df[~result_df["is_none_left"]]["left_true_id"]).mean()
    acc_ignore_none_right = (result_df[~result_df["is_none_right"]]["right_pred_id"] == result_df[~result_df["is_none_right"]]["right_true_id"]).mean()

    acc_ignore_none = (acc_ignore_none_left + acc_ignore_none_right) / 2
    print(acc_ignore_none)

    results = trainer.predict(lit_model, dm)
    logits_left = np.concatenate([result[0] for result in results], axis=0)
    logits_right = np.concatenate([result[1] for result in results], axis=0)
    pred_ids_left = np.argmax(logits_left, axis=1)
    pred_ids_right = np.argmax(logits_right, axis=1)

    result_data = []
    for i in range(len(dm.pred_ds)):
        result_data.append({
            "video_filename": dm.pred_ds.data[i]["video_filename"],
            "frame_filename": dm.pred_ds.data[i]["frame_filename"],
            "frame_idx": dm.pred_ds.data[i]["frame_idx"],
        })
    result_df = pd.DataFrame(result_data)
    action_to_id = {v: k for k, v in CFG.metadata["action_to_id"].items()} | {0: "none"}
    result_df["left_pred_action_id"] = pred_ids_left
    result_df["right_pred_action_id"] = pred_ids_right
    result_df["left_pred_action"] = result_df["left_pred_action_id"].map(action_to_id)
    result_df["right_pred_action"] = result_df["right_pred_action_id"].map(action_to_id)

    result_df.to_csv("preds.csv", index=False)
