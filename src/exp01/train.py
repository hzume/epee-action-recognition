
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
import albumentations as A
from tqdm import tqdm


class CFG:
    data_dir = Path("input/data_10hz")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1

    time_window = 1


def preprocess(df_orig: pd.DataFrame) -> pd.DataFrame:
    df = df_orig.copy()
    df["bbox_area"] = (df["bbox_x_2"] - df["bbox_x_1"]) * (df["bbox_y_2"] - df["bbox_y_1"])
    df["area"] = df["height"] * df["width"]
    df["bbox_area_ratio"] = df["bbox_area"] / df["area"]
    keypoint_score_cols = [col for col in df.columns if col.startswith("keypoint_score_")]
    all_keypoint_exits = df[keypoint_score_cols].min(axis=1) > 0.2
    is_in = (df["bbox_x_1"] > 0) & (df["bbox_x_2"] < df["width"]) & (df["bbox_y_1"] > 0) & (df["bbox_y_2"] < df["height"])
    df = df[(df["bbox_area_ratio"] > 0.05) & (df["bbox_area_ratio"] < 0.4) & all_keypoint_exits & is_in]
    num_instances = df.groupby("frame_filename")["instance_id"].nunique()
    frame_filenames = num_instances[num_instances==2].index.to_list()
    df = df[df["frame_filename"].isin(frame_filenames)]
    

    rows = {
        "frame_filename": [],
        "video_filename": [],
        "frame_idx": [],
        "width": [],
        "height": [],
        "side": [],
        "action": [],
    } | {f"keypoint_{i}_x": [] for i in range(17)} \
        | {f"keypoint_{i}_y": [] for i in range(17)} \
        | {f"keypoint_{i}_x_opposite": [] for i in range(17)} \
        | {f"keypoint_{i}_y_opposite": [] for i in range(17)} \

    for frame_filename in tqdm(frame_filenames, total=len(frame_filenames)):
        assert len(df[df["frame_filename"] == frame_filename]) == 2
        row1 = df[df["frame_filename"] == frame_filename].iloc[0]
        row2 = df[df["frame_filename"] == frame_filename].iloc[1]

        center_1 = (row1["bbox_x_1"] + row1["bbox_x_2"]) / 2
        center_2 = (row2["bbox_x_1"] + row2["bbox_x_2"]) / 2
        side_1 = "left" if center_1 < center_2 else "right"
        side_2 = "left" if center_2 < center_1 else "right"

        action_side = "left" if (row1["label"].split("_")[0] == "l") else "right"

        rows["frame_filename"].append(frame_filename)
        rows["video_filename"].append(row1["video_filename"])
        rows["frame_idx"].append(row1["frame_idx"])
        rows["width"].append(row1["width"])
        rows["height"].append(row1["height"])
        rows["side"].append(side_1)
        if side_1 == action_side:
            rows["action"].append(row1["action"])
        else:
            rows["action"].append(np.nan)
        for i in range(17):
            rows[f"keypoint_{i}_x"].append(row1[f"keypoint_{i}_x"])
            rows[f"keypoint_{i}_y"].append(row1[f"keypoint_{i}_y"])
            rows[f"keypoint_{i}_x_opposite"].append(row2[f"keypoint_{i}_x"])
            rows[f"keypoint_{i}_y_opposite"].append(row2[f"keypoint_{i}_y"])

        rows["frame_filename"].append(frame_filename)
        rows["video_filename"].append(row2["video_filename"])
        rows["frame_idx"].append(row2["frame_idx"])
        rows["width"].append(row2["width"])
        rows["height"].append(row2["height"])
        rows["side"].append(side_2)
        rows["action"].append(row2["action"])

    return pd.DataFrame(rows)


if __name__ == "__main__":
    pose_df = pd.read_csv(CFG.data_dir / "pose_preds.csv")
    pose_df = preprocess(pose_df)
    print(pose_df.head())
