import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from sklearn.metrics import f1_score


switch_side = {
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

class CFG:
    data_dir = Path("input/data_10hz")
    with open(data_dir.parent / "metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1
    predict_videos = ['2024-11-10-18-33-49.mp4', '2024-11-10-19-21-45.mp4', '2025-01-04_08-37-18.mp4', '2025-01-04_08-40-12.mp4']


def prepare_label_df(df: pd.DataFrame, action_to_id: dict[str, int]) -> pd.DataFrame:
    df["bbox_area"] = (df["bbox_x_2"] - df["bbox_x_1"]) * (df["bbox_y_2"] - df["bbox_y_1"])
    df["bbox_ratio"] = df["bbox_area"] / (df["width"] * df["height"])
    df["min_keypoint_score"] = df[[f"keypoint_score_{i}" for i in range(17)]].min(axis=1)
    df["center_x"] = (df["bbox_x_1"] + df["bbox_x_2"]) / 2
    df["center_y"] = (df["bbox_y_1"] + df["bbox_y_2"]) / 2
    df = df[(df["min_keypoint_score"] > 0.2) & (df["bbox_ratio"] > 0.0375) & (df["bbox_ratio"] < 0.15)]

    data = {
        "frame_filename": [],
        "video_filename": [],
        "frame_idx": [],
        "height": [],
        "width": [],
        "left_action_id": [],
        "right_action_id": [],
    } | {f"left_keypoint_{i}_x": [] for i in range(17)} \
    | {f"left_keypoint_{i}_y": [] for i in range(17)} \
    | {f"right_keypoint_{i}_x": [] for i in range(17)} \
    | {f"right_keypoint_{i}_y": [] for i in range(17)} \
    | {"left_center_x": [], "left_center_y": [], "right_center_x": [], "right_center_y": []} \
    | {"left_bbox_area": [], "right_bbox_area": []}
    
    for frame_filename, df_frame in df.groupby("frame_filename"):
        if len(df_frame) == 1:
            continue
        data["frame_filename"].append(frame_filename)
        data["video_filename"].append(df_frame["video_filename"].iloc[0])
        data["frame_idx"].append(df_frame["frame_idx"].iloc[0])
        data["height"].append(df_frame["height"].iloc[0])
        data["width"].append(df_frame["width"].iloc[0])
        data["left_action_id"].append(action_to_id[df_frame["left_action"].iloc[0]] if (df_frame["left_action"].iloc[0] in action_to_id) else 0)
        data["right_action_id"].append(action_to_id[df_frame["right_action"].iloc[0]] if (df_frame["right_action"].iloc[0] in action_to_id) else 0)

        target_rows = df_frame.sort_values(by="min_keypoint_score", ascending=False)[:2]
        
        if target_rows["center_x"].iloc[0] < target_rows["center_x"].iloc[1]:
            left_row = target_rows.iloc[0]
            right_row = target_rows.iloc[1]
        else:
            left_row = target_rows.iloc[1]
            right_row = target_rows.iloc[0]
            
        for i in range(17):
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

    pose_df = pd.DataFrame(data)
    
    for i in range(17):
        pose_df[f"left_keypoint_{i}_x"] = (pose_df[f"left_keypoint_{i}_x"] - pose_df["left_center_x"]) / pose_df["left_bbox_area"]
        pose_df[f"left_keypoint_{i}_y"] = (pose_df[f"left_keypoint_{i}_y"] - pose_df["left_center_y"]) / pose_df["left_bbox_area"]
        pose_df[f"right_keypoint_{i}_x"] = (pose_df[f"right_keypoint_{i}_x"] - pose_df["right_center_x"]) / pose_df["right_bbox_area"]
        pose_df[f"right_keypoint_{i}_y"] = (pose_df[f"right_keypoint_{i}_y"] - pose_df["right_center_y"]) / pose_df["right_bbox_area"]

    pose_df["distance"] = (pose_df["right_center_x"] - pose_df["left_center_x"]) / (pose_df["right_bbox_area"] + pose_df["left_bbox_area"])


    # switch side
    pose_df_switched = pose_df.copy()

    pose_df_switched["left_action_id"] = pose_df["right_action_id"]
    pose_df_switched["right_action_id"] = pose_df["left_action_id"]

    for i in range(17):
        j = switch_side[i]
        pose_df_switched[f"left_keypoint_{i}_x"] = -pose_df[f"right_keypoint_{j}_x"]
        pose_df_switched[f"left_keypoint_{i}_y"] = pose_df[f"right_keypoint_{j}_y"]
        pose_df_switched[f"right_keypoint_{i}_x"] = -pose_df[f"left_keypoint_{j}_x"]
        pose_df_switched[f"right_keypoint_{i}_y"] = pose_df[f"left_keypoint_{j}_y"]
    
    pose_df_switched["distance"] = pose_df_switched["distance"]
        
    pose_df = pd.concat([pose_df, pose_df_switched])

    return pose_df


if __name__ == "__main__":
    df = pd.read_csv(CFG.data_dir / "pose_preds.csv")
    df = prepare_label_df(df, CFG.metadata["action_to_id"])

    kp_idxs = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    feature_cols = [f"left_keypoint_{i}_x" for i in kp_idxs] + [f"left_keypoint_{i}_y" for i in kp_idxs] + [f"right_keypoint_{i}_x" for i in kp_idxs] + [f"right_keypoint_{i}_y" for i in kp_idxs] + ["distance"]
    # feature_cols = [f"right_keypoint_{i}_x" for i in kp_idxs] + [f"right_keypoint_{i}_y" for i in kp_idxs]
    
    pred_df = df[df["video_filename"].isin(CFG.predict_videos)]
    pred_df = pred_df.iloc[:len(pred_df) // 2]
    df = df[~df["video_filename"].isin(CFG.predict_videos)]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros((len(df), CFG.num_classes))
    models = []
    for train_idx, valid_idx in skf.split(df, df["right_action_id"], groups=df["video_filename"]):
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]

        train_df_neg = train_df[train_df["right_action_id"] == 0]
        train_df_pos = train_df[train_df["right_action_id"] != 0]

        train_df_neg = train_df_neg.sample(n=len(train_df_pos))
        train_df = pd.concat([train_df_neg, train_df_pos])

        print(train_df.shape, valid_df.shape)

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=CFG.num_classes,
            metric="multi_logloss",
            min_data_in_leaf=5,
            random_state=42,
            verbose=1
        )

        model.fit(train_df[feature_cols], train_df["right_action_id"], eval_set=[(valid_df[feature_cols], valid_df["right_action_id"])])
        oof[valid_idx] = model.predict_proba(valid_df[feature_cols])

        models.append(model)

    action_to_id = {v: k for k, v in CFG.metadata["action_to_id"].items()} | {0: "none"}
    df["pred_id"] = oof.argmax(axis=1)
    df["pred"] = [action_to_id[i] for i in df["pred_id"]]

    df["true_none"] = (df["right_action_id"] == 0)
    df["pred_none"] = (df["pred_id"] == 0)

    print((df["true_none"] == df["pred_none"]).mean())
    print((df[~df["true_none"]]["right_action_id"] == df[~df["true_none"]]["pred_id"]).mean())

    print(f1_score(df["right_action_id"], df["pred_id"], average="macro"))

    preds = np.zeros((len(pred_df), CFG.num_classes))
    for model in models:
        preds += model.predict_proba(pred_df[feature_cols])

    pred_df["right_pred_action_id"] = preds.argmax(axis=1)

    pred_df["right_pred_action"] = [action_to_id[i] for i in pred_df["right_pred_action_id"]]

    pred_df.to_csv("preds.csv", index=False)
