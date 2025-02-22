from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import json
import numpy as np
import torch
import albumentations as A
import random

def get_max_size(frame_label_df: pd.DataFrame, frame_dir: Path) -> tuple[int, int]:
    max_height = 0
    max_width = 0
    for video_filename in frame_label_df["video_filename"].unique():
        video_filename = video_filename.replace(".mp4", "")
        frame_path = frame_dir / f"{video_filename}_0.jpg"
        img = cv2.imread(frame_path)
        max_height = max(max_height, img.shape[0])
        max_width = max(max_width, img.shape[1])
    return max_height, max_width


def resize_and_pad(
    img: np.ndarray, target_size: tuple[int, int], padding_color=(0, 0, 0)
):
    orig_height, orig_width = img.shape[:2]
    target_height, target_width = target_size

    # アスペクト比を保ちながらリサイズするスケールを計算
    scale_factor = min(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # リサイズ
    resized_img = cv2.resize(
        img, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )

    # パディングサイズを計算 (上下左右)
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left

    # パディング (指定した色で埋める)
    padded_img = cv2.copyMakeBorder(
        resized_img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color,
    )

    return padded_img


def calculate_label_prob_for_video(df_video: pd.DataFrame, width: int, alpha: float = 0.5) -> pd.DataFrame:
    """
    1) labelが NaN でなければ label_prob = 1
    2) NaN の場合、前後3フレームに NaN でない label があれば、
       (0.5)^(そのフレームまでの frame_id 差) を label_prob とする
    3) 前後3フレームに NaN でない label が一つも無い場合は label_prob = 0
    """
    # frame_id 昇順にソート（念のため）
    df_video = df_video.sort_values("frame_idx").reset_index(drop=True)
    
    frame_idxs = df_video["frame_idx"].values
    n = len(df_video)
    
    def calculate_label_prob(
        action_ids: list[list[int]], 
    ) -> tuple[list[int], list[float]]:
        action_ids_list = []
        label_probs_list = []
        for i in range(n):
            action_ids_next = []
            label_probs_next = []
            
            # 2) NaN の場合、前後3フレーム内に NaN でない label があるか探索
            start_idx = max(0, i - width)
            end_idx = min(n, i + width + 1)  # i+3 まで含めるため +4
            distances = {}
            
            for j in range(start_idx, end_idx):
                # ラベルが存在するフレームを見つけたら距離を計算
                for action_id in action_ids[j]:
                    if action_id != 0:
                        dist = abs(frame_idxs[i] - frame_idxs[j])
                        if action_id not in distances:
                            distances[action_id] = dist
                        else:
                            distances[action_id] = min(distances[action_id], dist)
            
            # 距離の中で最小のものを使って (0.5)^dist
            if len(distances) == 0:
                action_ids_next.append(0)
                label_probs_next.append(1.0)
            else:
                for action_id, dist in distances.items():
                    if action_id not in action_ids_next:
                        action_ids_next.append(action_id)
                        label_probs_next.append(alpha ** dist)
            assert (len(action_ids_next) == len(label_probs_next))
            action_ids_list.append(action_ids_next)
            label_probs_list.append(label_probs_next)
        assert (len(action_ids_list) == n)
        assert (len(label_probs_list) == n)
        return action_ids_list, label_probs_list
    
    left_action_ids_list, left_label_probs_list = calculate_label_prob(df_video["left_action_ids"].values)
    right_action_ids_list, right_label_probs_list = calculate_label_prob(df_video["right_action_ids"].values)

    df_video["left_action_ids"] = left_action_ids_list
    df_video["left_label_probs"] = left_label_probs_list
    df_video["right_action_ids"] = right_action_ids_list
    df_video["right_label_probs"] = right_label_probs_list
    return df_video


def preprocess_frame_label_df(df: pd.DataFrame, time_window: int, width: int, frame_dir: Path, action_to_id: dict[str, int]) -> pd.DataFrame:

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
        df["frame_idx_min"] + time_window,
        df["frame_idx_max"] - time_window,
    )
    df = df[is_in_time_window]
    frame_paths = []
    for _, row in df.iterrows():
        frame_idx = row["frame_idx"]
        video_basename = row["video_filename"].replace(".mp4", "")
        frame_idxs = list(
            range(frame_idx - time_window, frame_idx + time_window + 1)
        )
        frame_paths.append(
            [
                str(frame_dir / f"{video_basename}_{frame_idx}.jpg")
                for frame_idx in frame_idxs
            ]
        )
    df["frame_paths"] = frame_paths

    df = df.groupby("video_filename").apply(calculate_label_prob_for_video, width=width)
    return df


def sample_negative_frames(df: pd.DataFrame) -> pd.DataFrame:
    df_pos = df[(df["left_action_ids"].apply(lambda x: x != [0]) | (df["right_action_ids"].apply(lambda x: x != [0])))]
    df_neg = df[(df["left_action_ids"].apply(lambda x: x == [0]) & (df["right_action_ids"].apply(lambda x: x == [0])))]
    df_neg = df_neg.sample(n=len(df_pos) // 5, replace=True)
    df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    return df


class Dataset1(Dataset):
    def __init__(
        self,
        frame_dir: Path,
        frame_label_df: pd.DataFrame,
        time_window: int,
        width: int,
        max_width: int,
        max_height: int,
        action_to_id: dict[str, int],
        num_classes: int,
        transforms: Optional[Callable] = None,
        sampling: bool = True,
        p_flip: float = 0.5,
    ):
        self.frame_label_df = preprocess_frame_label_df(
            df=frame_label_df,
            time_window=time_window,
            width=width,
            frame_dir=frame_dir,
            action_to_id=action_to_id,
        )
        if sampling:
            self.frame_label_df = sample_negative_frames(self.frame_label_df)
        self.transforms = transforms
        self.max_width = max_width
        self.max_height = max_height
        self.num_classes = num_classes
        self.p_flip = p_flip

    def __len__(self):
        return len(self.frame_label_df)

    def __getitem__(self, idx: int):
        flip = random.random() < self.p_flip
        frame_paths = self.frame_label_df.iloc[idx]["frame_paths"]
        left_action_ids = self.frame_label_df.iloc[idx]["left_action_ids"]
        right_action_ids = self.frame_label_df.iloc[idx]["right_action_ids"]
        left_label_probs = self.frame_label_df.iloc[idx]["left_label_probs"]
        right_label_probs = self.frame_label_df.iloc[idx]["right_label_probs"]
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frame = resize_and_pad(frame, (self.max_height, self.max_width))
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        if self.transforms:
            replay = self.transforms(image=frames[0])["replay"]
            frames = [A.ReplayCompose.replay(replay, image=frame)["image"] for frame in frames]
            # frames = [self.transforms(image=frame)["image"] for frame in frames]
        frames = np.stack(frames)
        
        left_label = torch.zeros(self.num_classes)
        right_label = torch.zeros(self.num_classes)
        for action_id, label_prob in zip(left_action_ids, left_label_probs):
            left_label[action_id] = label_prob
        for action_id, label_prob in zip(right_action_ids, right_label_probs):
            right_label[action_id] = label_prob

        if flip:
            frames = frames[:, :, ::-1, :].copy()
            left_label, right_label = right_label, left_label
        return frames, left_label.float(), right_label.float()


if __name__ == "__main__":
    data_dir = Path("input/data_10hz")
    df = pd.read_csv(data_dir / "frame_label.csv")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    dataset = Dataset1(
        frame_dir=data_dir / "frames",
        frame_label_df=df,
        time_window=2,
        width=4,
        max_width=224,
        max_height=224,
        action_to_id=metadata["action_to_id"],
        num_classes=num_classes,
        sampling=True,
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    for x, y_left, y_right in dataloader:
        print(x.shape)
        print(y_left)
        print(y_right)
        break
