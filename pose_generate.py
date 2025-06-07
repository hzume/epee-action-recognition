from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

from mmpose.apis import MMPoseInferencer

data_dir = Path("input/data_10hz")
preds_dir = Path("output/pose_10hz_3d")

frame_label_df = pd.read_csv(data_dir / "frame_label.csv")
frame_filename_paths = [str(data_dir / "frames" / frame_filename) for frame_filename in frame_label_df["frame_filename"]]

processed_frame_paths = [str(data_dir / "frames" / pred_path.stem) + ".jpg" for pred_path in preds_dir.glob("*.json")]
frame_filename_paths = [frame_path for frame_path in frame_filename_paths if frame_path not in processed_frame_paths]

inferencer = MMPoseInferencer(pose3d="human3d")

result_generator = inferencer(frame_filename_paths, pred_out_dir=str(preds_dir), return_vis=False)

progbar = tqdm(total=len(frame_filename_paths), desc="Processing frames...")
for result in result_generator:
    batch_size = len(result["predictions"])
    progbar.update(batch_size)
    progbar.set_postfix({"batch_size": batch_size})
