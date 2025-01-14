from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import numpy as np

def resize_and_pad(img: np.ndarray, target_size: tuple[int, int], padding_color=(0, 0, 0)):
    """
    OpenCVを使ってアスペクト比を保ちながら画像をリサイズし、足りない部分をパディングする関数。

    Args:
        img (numpy.ndarray): 入力画像（BGR形式）。
        target_size (tuple): (height, width) 形式のターゲットサイズ。
        padding_color (tuple): パディングに使用する色 (B, G, R)。

    Returns:
        padded_img (numpy.ndarray): リサイズ＆パディング後の画像。
    """
    orig_height, orig_width = img.shape[:2]
    target_height, target_width = target_size

    # アスペクト比を保ちながらリサイズするスケールを計算
    scale_factor = min(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # リサイズ
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # パディングサイズを計算 (上下左右)
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left

    # パディング (指定した色で埋める)
    padded_img = cv2.copyMakeBorder(
        resized_img, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_color
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
        transforms: Optional[Callable] = None,
    ):
        self.frame_label_df = frame_label_df.copy()
        self.transforms = transforms
        self.max_width = max_width
        self.max_height = max_height

        frame_min_idx = (
            self.frame_label_df.groupby("video_filename")["frame_idx"]
            .min()
            .reset_index(name="frame_idx_min")
        )
        frame_max_idx = (
            self.frame_label_df.groupby("video_filename")["frame_idx"]
            .max()
            .reset_index(name="frame_idx_max")
        )
        self.frame_label_df = self.frame_label_df.merge(
            frame_min_idx, on="video_filename"
        )
        self.frame_label_df = self.frame_label_df.merge(
            frame_max_idx, on="video_filename"
        )

        is_in_time_window = self.frame_label_df["frame_idx"].between(
            self.frame_label_df["frame_idx_min"] + time_window,
            self.frame_label_df["frame_idx_max"] - time_window,
        )
        self.frame_label_df = self.frame_label_df[is_in_time_window]
        frame_paths = []
        for _, row in self.frame_label_df.iterrows():
            frame_idx = row["frame_idx"]
            video_basename = row["video_filename"].replace(".mp4", "")
            frame_idxs = list(range(frame_idx - time_window, frame_idx + time_window + 1))
            frame_paths.append([str(frame_dir / f"{video_basename}_{frame_idx}.jpg") for frame_idx in frame_idxs])
        self.frame_label_df["frame_paths"] = frame_paths

    def __len__(self):
        return len(self.frame_label_df)

    def __getitem__(self, idx: int):
        frame_paths = self.frame_label_df.iloc[idx]["frame_paths"]
        frames = [cv2.imread(frame_path) for frame_path in frame_paths]
        frames = [resize_and_pad(frame, (self.max_height, self.max_width)) for frame in frames]
        if self.transforms:
            frames = [self.transforms(frame) for frame in frames]
        frames = np.stack(frames)
        label = self.frame_label_df.iloc[idx]["label"]
        return frames


if __name__ == "__main__":
    frame_label_df = pd.read_csv(Path("input/data/frame_label.csv"))
    dataset = Dataset0(
        frame_dir=Path("input/data/frames"),
        frame_label_df=frame_label_df,
        time_window=2,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for frames in loader:
        for batch_idx in range(frames.shape[0]):
            img = frames[batch_idx, 0, :, :, :]
            # torch.Tensor -> numpy.ndarray
            img = img.numpy()
            cv2.imwrite(f"output/frame_{batch_idx}.jpg", img)
        break


