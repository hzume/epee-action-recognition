from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import json
import numpy as np
import torch

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


class Dataset0(Dataset):
    def __init__(
        self,
        frame_dir: Path,
        frame_label_df: pd.DataFrame,
        time_window: int,
        max_width: int,
        max_height: int,
        action_to_id: dict[str, int],
        num_classes: int,
        transforms: Optional[Callable] = None,
    ):
        self.frame_label_df = frame_label_df
        self.transforms = transforms
        self.max_width = max_width
        self.max_height = max_height
        self.num_classes = num_classes

    def __len__(self):
        return len(self.frame_label_df)

    def __getitem__(self, idx: int):
        frame_paths = self.frame_label_df.iloc[idx]["frame_paths"]
        label_prob = self.frame_label_df.iloc[idx]["label_prob"]
        frames = []
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            frame = resize_and_pad(frame, (self.max_height, self.max_width))
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        if self.transforms:
            frames = [self.transforms(frame) for frame in frames]
        frames = np.stack(frames)
        label = self.frame_label_df.iloc[idx]["action_id"]
        label_tensor = torch.tensor(label, dtype=torch.int64)
        label_one_hot = one_hot(label_tensor, num_classes=self.num_classes)
        if label != 0:
            label_one_hot = label_one_hot * label_prob
        return frames, label_one_hot.float()


if __name__ == "__main__":
    data_dir = Path("input/data")
    frame_label_df = pd.read_csv(data_dir / "frame_label.csv")
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    max_height, max_width = get_max_size(frame_label_df, data_dir / "frames")

    dataset = Dataset0(
        frame_dir=data_dir / "frames",
        frame_label_df=frame_label_df,
        time_window=2,
        max_height=max_height,
        max_width=max_width,
        action_to_id=metadata["action_to_id"],
        num_classes=num_classes,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for frames, label in loader:
        print(frames.shape)
        print(label)
        break
