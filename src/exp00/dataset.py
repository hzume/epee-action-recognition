from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class Dataset0(Dataset):
    def __init__(self, frame_dir: Path, frame_label_df: pd.DataFrame, time_window: int, transforms: Optional[Callable] = None):
        all_frames = []
        for frame_filename in tqdm(frame_label_df["frame_filename"], desc="loading..."):
            frame_path = frame_dir / frame_filename
            frame = cv2.imread(frame_path)
            all_frames.append(frame)
        self.all_frames = all_frames

    def __len__(self):
        return len(self.all_frames)


if __name__ == "__main__":
    frame_label_df = pd.read_csv(Path("input/data/frame_label.csv"))
    print("aa")
    dataset = Dataset0(frame_dir=Path("input/data/frames"), frame_label_df=frame_label_df, time_window=10)
    print(len(dataset))
    print(dataset[0].shape)
