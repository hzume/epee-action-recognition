import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

data_dir = Path("input/data_10hz")
preds_dir = Path("output/pose_10hz")

frame_label_df = pd.read_csv(data_dir / "frame_label.csv")

pose_data = {
    "frame_filename": [],
    "video_filename": [],
    "frame_idx": [],
    "label": [],
    "action": [],
    "instance_id": [],
} | {f"keypoint_{i}_x": [] for i in range(17)} \
    | {f"keypoint_{i}_y": [] for i in range(17)} \
    | {f"keypoint_score_{i}": [] for i in range(17)} \
    | {"bbox_x_1": [], "bbox_y_1": [], "bbox_x_2": [], "bbox_y_2": [], "bbox_score": []}

for _, row in tqdm(frame_label_df.iterrows(), total=len(frame_label_df)):
    frame_filename = row["frame_filename"]
    frame_basename = Path(frame_filename).stem
    pred_path = preds_dir / f"{frame_basename}.json"
    with open(pred_path, "r") as f:
        preds = json.load(f)
    
    for i, instance in enumerate(preds):
        keypoints = instance["keypoints"]
        keypoint_scores = instance["keypoint_scores"]
        bbox = instance["bbox"][0]
        bbox_score = instance["bbox_score"]

        pose_data["frame_filename"].append(frame_filename)
        pose_data["video_filename"].append(row["video_filename"])
        pose_data["frame_idx"].append(row["frame_idx"])
        pose_data["label"].append(row["label"])
        pose_data["action"].append(row["action"])
        pose_data["instance_id"].append(i)

        for j, keypoint in enumerate(keypoints):
            pose_data[f"keypoint_{j}_x"].append(keypoint[0])
            pose_data[f"keypoint_{j}_y"].append(keypoint[1])
            pose_data[f"keypoint_score_{j}"].append(keypoint_scores[j])
        
        pose_data["bbox_x_1"].append(bbox[0])
        pose_data["bbox_y_1"].append(bbox[1])
        pose_data["bbox_x_2"].append(bbox[2])
        pose_data["bbox_y_2"].append(bbox[3])
        pose_data["bbox_score"].append(bbox_score)

pose_df = pd.DataFrame(pose_data)
print(pose_df.head())
pose_df.to_csv(data_dir / "pose_preds.csv", index=False)