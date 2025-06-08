import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
from sklearn.metrics import f1_score
import argparse
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

class CFG:
    data_dir = Path("input/data_10hz_3d")
    with open("input/metadata.json", "r") as f:
        metadata = json.load(f)
    num_classes = len(metadata["action_to_id"]) + 1
    predict_videos = ['2024-11-10-18-27-14.mp4', '2025-01-04_08-15-17.mp4']


def prepare_label_df(df: pd.DataFrame, action_to_id: dict[str, int]) -> pd.DataFrame:
    """3D poseデータから特徴量を作成"""
    # フレームごとに2つのインスタンスを持つデータのみを使用
    data = {
        "frame_filename": [],
        "video_filename": [],
        "frame_idx": [],
        "left_action_id": [],
        "right_action_id": [],
    }
    
    # 3Dキーポイント座標を追加
    for i in range(17):
        data[f"left_keypoint_{i}_x"] = []
        data[f"left_keypoint_{i}_y"] = []
        data[f"left_keypoint_{i}_z"] = []
        data[f"right_keypoint_{i}_x"] = []
        data[f"right_keypoint_{i}_y"] = []
        data[f"right_keypoint_{i}_z"] = []
    
    # フレームごとにグループ化
    for (video_filename, frame_idx), df_frame in df.groupby(["video_filename", "frame_idx"]):
        if len(df_frame) != 2:
            continue  # ちょうど2つのインスタンスがあるフレームのみ使用
        
        # インスタンスを左右に分類（x座標で判定）
        df_frame = df_frame.sort_values("keypoint_0_x")  # pelvisのx座標でソート
        left_row = df_frame.iloc[0]
        right_row = df_frame.iloc[1]
        
        data["frame_filename"].append(f"{video_filename}_{frame_idx}")
        data["video_filename"].append(video_filename)
        data["frame_idx"].append(frame_idx)
        data["left_action_id"].append(action_to_id.get(left_row["left_action"], 0))
        data["right_action_id"].append(action_to_id.get(right_row["right_action"], 0))
        
        # 3Dキーポイント座標を追加
        for i in range(17):
            data[f"left_keypoint_{i}_x"].append(left_row[f"keypoint_{i}_x"])
            data[f"left_keypoint_{i}_y"].append(left_row[f"keypoint_{i}_y"])
            data[f"left_keypoint_{i}_z"].append(left_row[f"keypoint_{i}_z"])
            data[f"right_keypoint_{i}_x"].append(right_row[f"keypoint_{i}_x"])
            data[f"right_keypoint_{i}_y"].append(right_row[f"keypoint_{i}_y"])
            data[f"right_keypoint_{i}_z"].append(right_row[f"keypoint_{i}_z"])
    
    pose_df = pd.DataFrame(data)
    
    # 正規化（pelvis中心の相対座標に変換）
    for i in range(17):
        # 左のプレイヤー
        pose_df[f"left_keypoint_{i}_x"] = pose_df[f"left_keypoint_{i}_x"] - pose_df["left_keypoint_0_x"]
        pose_df[f"left_keypoint_{i}_y"] = pose_df[f"left_keypoint_{i}_y"] - pose_df["left_keypoint_0_y"]
        pose_df[f"left_keypoint_{i}_z"] = pose_df[f"left_keypoint_{i}_z"] - pose_df["left_keypoint_0_z"]
        
        # 右のプレイヤー
        pose_df[f"right_keypoint_{i}_x"] = pose_df[f"right_keypoint_{i}_x"] - pose_df["right_keypoint_0_x"]
        pose_df[f"right_keypoint_{i}_y"] = pose_df[f"right_keypoint_{i}_y"] - pose_df["right_keypoint_0_y"]
        pose_df[f"right_keypoint_{i}_z"] = pose_df[f"right_keypoint_{i}_z"] - pose_df["right_keypoint_0_z"]
    
    # プレイヤー間の距離
    pose_df["distance"] = np.sqrt((pose_df["right_keypoint_0_x"] - pose_df["left_keypoint_0_x"])**2 + 
                                 (pose_df["right_keypoint_0_y"] - pose_df["left_keypoint_0_y"])**2 + 
                                 (pose_df["right_keypoint_0_z"] - pose_df["left_keypoint_0_z"])**2)
    
    pose_df = add_pose_features(pose_df)
    return pose_df


def add_pose_features(pose_df: pd.DataFrame) -> pd.DataFrame:
    """3Dポーズ特徴量を追加"""
    # Create a copy to avoid fragmentation warnings
    pose_df = pose_df.copy()
    
    # Prepare feature dictionaries for batch addition
    new_features = {}
    
    # 各キーポイントの距離と角度
    for player in ["left", "right"]:
        for i in range(17):
            # ルートからの距離
            new_features[f"{player}_keypoint_{i}_dist"] = np.sqrt(
                pose_df[f"{player}_keypoint_{i}_x"] ** 2 + 
                pose_df[f"{player}_keypoint_{i}_y"] ** 2 + 
                pose_df[f"{player}_keypoint_{i}_z"] ** 2
            )
            
            # 角度（水平面での角度）
            new_features[f"{player}_keypoint_{i}_angle_xy"] = np.arctan2(
                pose_df[f"{player}_keypoint_{i}_y"], 
                pose_df[f"{player}_keypoint_{i}_x"]
            )
            
            # 角度（垂直面での角度）
            new_features[f"{player}_keypoint_{i}_angle_xz"] = np.arctan2(
                pose_df[f"{player}_keypoint_{i}_z"], 
                pose_df[f"{player}_keypoint_{i}_x"]
            )
    
    # 関節角度（Human3Dキーポイント構造に基づく）
    joint_pairs = [
        # 左腕
        ("left_shoulder", 11, 12),  # 左肩-左肘
        ("left_elbow", 12, 13),     # 左肘-左手首
        # 右腕
        ("right_shoulder", 14, 15), # 右肩-右肘
        ("right_elbow", 15, 16),    # 右肘-右手首
        # 左脚
        ("left_hip", 1, 2),         # 左腰-左膝
        ("left_knee", 2, 3),        # 左膝-左足首
        # 右脚
        ("right_hip", 4, 5),        # 右腰-右膝
        ("right_knee", 5, 6),       # 右膝-右足首
        # 胴体
        ("spine", 0, 7),            # pelvis-spine
        ("torso", 7, 8),            # spine-thorax
        ("neck", 9, 10),            # neck-head
    ]
    
    for player in ["left", "right"]:
        for joint_name, point1, point2 in joint_pairs:
            # ベクトルを計算
            dx = pose_df[f"{player}_keypoint_{point2}_x"] - pose_df[f"{player}_keypoint_{point1}_x"]
            dy = pose_df[f"{player}_keypoint_{point2}_y"] - pose_df[f"{player}_keypoint_{point1}_y"]
            dz = pose_df[f"{player}_keypoint_{point2}_z"] - pose_df[f"{player}_keypoint_{point1}_z"]
            
            # 長さ
            new_features[f"{player}_{joint_name}_length"] = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 角度
            new_features[f"{player}_{joint_name}_angle_xy"] = np.arctan2(dy, dx)
            new_features[f"{player}_{joint_name}_angle_xz"] = np.arctan2(dz, dx)
    
    # Add all new features at once using concat
    new_features_df = pd.DataFrame(new_features, index=pose_df.index)
    pose_df = pd.concat([pose_df, new_features_df], axis=1)
    
    return pose_df


def switch_side(pose_df: pd.DataFrame) -> pd.DataFrame:
    """左右を入れ替えてデータ拡張"""
    pose_df_switched = pose_df.copy()
    
    # アクションIDを入れ替え
    pose_df_switched["left_action_id"] = pose_df["right_action_id"]
    pose_df_switched["right_action_id"] = pose_df["left_action_id"]
    
    # キーポイントを入れ替え（Human3Dキーポイント構造）
    keypoint_mirror_map = {
        0: 0,   # pelvis
        1: 4,   # left_hip -> right_hip
        2: 5,   # left_knee -> right_knee
        3: 6,   # left_ankle -> right_ankle
        4: 1,   # right_hip -> left_hip
        5: 2,   # right_knee -> left_knee
        6: 3,   # right_ankle -> left_ankle
        7: 7,   # spine
        8: 8,   # thorax
        9: 9,   # neck
        10: 10, # head
        11: 14, # left_shoulder -> right_shoulder
        12: 15, # left_elbow -> right_elbow
        13: 16, # left_wrist -> right_wrist
        14: 11, # right_shoulder -> left_shoulder
        15: 12, # right_elbow -> left_elbow
        16: 13, # right_wrist -> left_wrist
    }
    
    for i in range(17):
        j = keypoint_mirror_map[i]
        # X座標は反転、Y, Z座標はそのまま
        pose_df_switched[f"left_keypoint_{i}_x"] = -pose_df[f"right_keypoint_{j}_x"]
        pose_df_switched[f"left_keypoint_{i}_y"] = pose_df[f"right_keypoint_{j}_y"]
        pose_df_switched[f"left_keypoint_{i}_z"] = pose_df[f"right_keypoint_{j}_z"]
        
        pose_df_switched[f"right_keypoint_{i}_x"] = -pose_df[f"left_keypoint_{j}_x"]
        pose_df_switched[f"right_keypoint_{i}_y"] = pose_df[f"left_keypoint_{j}_y"]
        pose_df_switched[f"right_keypoint_{i}_z"] = pose_df[f"left_keypoint_{j}_z"]
    
    # 既存の特徴量カラムを削除してから再計算
    feature_cols_to_drop = [col for col in pose_df_switched.columns if 
                           col.endswith('_dist') or col.endswith('_angle_xy') or 
                           col.endswith('_angle_xz') or col.endswith('_length')]
    pose_df_switched = pose_df_switched.drop(columns=feature_cols_to_drop)
    
    # 特徴量を再計算
    pose_df_switched = add_pose_features(pose_df_switched)
    
    return pose_df_switched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="input/data_10hz_3d")
    parser.add_argument("--output", type=str, default="preds.csv")
    args = parser.parse_args()
    
    # データ読み込み
    CFG.data_dir = Path(args.data_dir)
    
    # 3D poseデータを読み込み
    pose_3d_file = CFG.data_dir / "train_3d.csv"
    if not pose_3d_file.exists():
        print(f"Error: {pose_3d_file} not found. Please run preprocessing first.")
        return
    
    df = pd.read_csv(pose_3d_file)
    print(f"Loaded {len(df)} samples from {pose_3d_file}")
    
    # データ準備
    df = prepare_label_df(df, CFG.metadata["action_to_id"])
    print(f"Prepared {len(df)} frame pairs")
    
    # 特徴量カラムを選択（角度、距離、長さ）
    feature_cols = [col for col in df.columns if 
                   ("action" not in col) and 
                   (col.endswith("angle_xy") or col.endswith("angle_xz") or 
                    col.endswith("dist") or col.endswith("length") or col == "distance")]
    
    print(f"Using {len(feature_cols)} features")
    
    # 予測用データ（テストビデオ）を分離
    pred_df = df[df["video_filename"].isin(CFG.predict_videos)]
    df = df[~df["video_filename"].isin(CFG.predict_videos)]
    
    # データ拡張（左右入れ替え）
    df_switched = switch_side(df)
    df = pd.concat([df, df_switched], ignore_index=True)
    print(f"After augmentation: {len(df)} samples")
    
    # クロスバリデーション
    skf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    oof = np.zeros((len(df), CFG.num_classes))
    models = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["left_action_id"], groups=df["video_filename"])):
        print(f"\n=== Fold {fold + 1} ===")
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[valid_idx]
        
        # クラスバランス調整（noneクラスをダウンサンプリング）
        train_df_neg = train_df[train_df["left_action_id"] == 0]
        train_df_pos = train_df[train_df["left_action_id"] != 0]
        
        if len(train_df_pos) > 0:
            train_df_neg = train_df_neg.sample(n=min(len(train_df_neg), len(train_df_pos)), random_state=42)
            train_df = pd.concat([train_df_neg, train_df_pos])
        
        print(f"Train: {train_df.shape}, Valid: {valid_df.shape}")
        print(f"Train class distribution: {train_df['left_action_id'].value_counts().to_dict()}")
        
        # LightGBMモデル
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=CFG.num_classes,
            metric="multi_logloss",
            num_leaves=31,
            max_depth=7,
            min_data_in_leaf=5,
            min_child_weight=1e-3,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=42,
            n_estimators=100,
            verbose=-1
        )
        
        # 訓練
        model.fit(
            train_df[feature_cols], 
            train_df["left_action_id"],
            eval_set=[(valid_df[feature_cols], valid_df["left_action_id"])],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # 予測
        oof[valid_idx] = model.predict_proba(valid_df[feature_cols])
        models.append(model)
        
        # 検証結果
        valid_pred = model.predict_proba(valid_df[feature_cols]).argmax(axis=1)
        valid_f1 = f1_score(valid_df["left_action_id"], valid_pred, average="macro")
        print(f"Fold {fold + 1} F1-score: {valid_f1:.4f}")
    
    # OOF評価
    action_to_id = {v: k for k, v in CFG.metadata["action_to_id"].items()}
    action_to_id[0] = "none"
    
    df["pred_id"] = oof.argmax(axis=1)
    df["pred"] = [action_to_id[i] for i in df["pred_id"]]
    
    # 全体の精度
    overall_acc = (df["left_action_id"] == df["pred_id"]).mean()
    overall_f1 = f1_score(df["left_action_id"], df["pred_id"], average="macro")
    
    print(f"\n=== Overall Results ===")
    print(f"Accuracy: {overall_acc:.4f}")
    print(f"F1-score (macro): {overall_f1:.4f}")
    
    # noneクラス以外の精度
    non_none_mask = df["left_action_id"] != 0
    if non_none_mask.sum() > 0:
        non_none_acc = (df[non_none_mask]["left_action_id"] == df[non_none_mask]["pred_id"]).mean()
        print(f"Non-none accuracy: {non_none_acc:.4f}")
    
    # 予測データがある場合
    if len(pred_df) > 0:
        print(f"\n=== Generating Predictions ===")
        
        # 左プレイヤーの予測
        left_preds = np.zeros((len(pred_df), CFG.num_classes))
        for model in models:
            left_preds += model.predict_proba(pred_df[feature_cols])
        left_preds /= len(models)
        
        # 右プレイヤーの予測（左右入れ替えたデータで予測）
        right_preds = np.zeros((len(pred_df), CFG.num_classes))
        pred_df_switched = switch_side(pred_df)
        for model in models:
            right_preds += model.predict_proba(pred_df_switched[feature_cols])
        right_preds /= len(models)
        
        pred_df["left_pred_action_id"] = left_preds.argmax(axis=1)
        pred_df["right_pred_action_id"] = right_preds.argmax(axis=1)
        pred_df["left_pred_action"] = pred_df["left_pred_action_id"].map(action_to_id)
        pred_df["right_pred_action"] = pred_df["right_pred_action_id"].map(action_to_id)
        
        # 確率スコアも追加
        for i, action_name in action_to_id.items():
            pred_df[f"left_pred_{action_name}_prob"] = left_preds[:, i]
            pred_df[f"right_pred_{action_name}_prob"] = right_preds[:, i]
        
        # 結果を保存
        output_cols = ["frame_filename", "video_filename", "frame_idx", 
                      "left_pred_action", "right_pred_action"]
        # 確率スコアのカラムも追加
        prob_cols = [col for col in pred_df.columns if col.endswith("_prob")]
        output_cols.extend(prob_cols)
        
        output_df = pred_df[output_cols].copy()
        output_df.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
        print(f"Saved {len(output_df)} predictions with {len(prob_cols)} probability scores")
        
        # 予測統計
        print(f"Left action distribution: {pred_df['left_pred_action'].value_counts().to_dict()}")
        print(f"Right action distribution: {pred_df['right_pred_action'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()