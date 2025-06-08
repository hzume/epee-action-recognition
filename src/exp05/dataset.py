"""Dataset module for exp05"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm

from .utils import Config


class EpeeDataset(Dataset):
    """Fencing action recognition dataset"""
    
    def __init__(
        self,
        raw_df: pd.DataFrame,
        config: Config,
        sample_balanced: bool = False,
        augment: bool = False,
        normalization_stats: Dict[str, np.ndarray] = None
    ):
        """
        Initialize dataset
        
        Args:
            raw_df: Raw dataframe with pose predictions
            config: Configuration object
            sample_balanced: Whether to balance positive/negative samples
            augment: Whether to apply data augmentation
            normalization_stats: Pre-computed normalization statistics (mean, std)
        """
        self.config = config
        self.action_to_id = config.metadata["action_to_id"]
        self.augment = augment
        self.normalization_stats = normalization_stats
        
        # Preprocess data
        self.df = self.preprocess(raw_df)
        
        # Define feature columns
        self.feature_cols = list(
            set(self.df.columns.tolist()) - {
                "frame_filename", "video_filename", "frame_idx",
                "left_action_id", "right_action_id", "switched"
            }
        )
        
        # Compute or use provided normalization statistics
        if self.normalization_stats is None:
            self.normalization_stats = self._compute_normalization_stats()
        
        # Create temporal windows
        self.data = self._create_temporal_windows()
        
        # Balance samples if requested
        if sample_balanced:
            self.data = self._balance_samples(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index"""
        item = self.data[idx]
        x = item["x"]
        
        # Apply augmentation if enabled
        if self.augment and random.random() < 0.5:
            x = self._augment(x)
        
        return x, item["y_left"], item["y_right"]
    
    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw dataframe"""
        # Process raw data
        df = self._process_raw_df(raw_df)
        
        # Create switched version for data augmentation
        df_switched = self._switch_side(df)
        
        df["switched"] = False
        df_switched["switched"] = True
        
        # Combine original and switched data
        df = pd.concat([df, df_switched], ignore_index=True)
        
        # Add pose features
        df = self._add_pose_features(df)
        
        return df
    
    def _process_raw_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Process raw dataframe to extract player poses"""
        # Calculate additional features
        raw_df["bbox_area"] = (raw_df["bbox_x_2"] - raw_df["bbox_x_1"]) * (raw_df["bbox_y_2"] - raw_df["bbox_y_1"])
        raw_df["bbox_ratio"] = raw_df["bbox_area"] / (raw_df["width"] * raw_df["height"])
        raw_df["min_keypoint_score"] = raw_df[[f"keypoint_score_{i}" for i in range(self.config.num_kp)]].min(axis=1)
        raw_df["center_x"] = (raw_df["bbox_x_1"] + raw_df["bbox_x_2"]) / 2
        raw_df["center_y"] = (raw_df["bbox_y_1"] + raw_df["bbox_y_2"]) / 2
        
        # Filter to extract target player bboxes
        raw_df = raw_df[
            (raw_df["min_keypoint_score"] > self.config.min_keypoint_score) &
            (raw_df["bbox_ratio"] > self.config.min_bbox_ratio) &
            (raw_df["bbox_ratio"] < self.config.max_bbox_ratio)
        ]
        
        # Initialize data dictionary
        data = {
            "frame_filename": [],
            "video_filename": [],
            "frame_idx": [],
            "left_action_id": [],
            "right_action_id": [],
        }
        
        # Add keypoint columns
        for side in ["left", "right"]:
            for i in range(self.config.num_kp):
                data[f"{side}_keypoint_{i}_x"] = []
                data[f"{side}_keypoint_{i}_y"] = []
            data[f"{side}_center_x"] = []
            data[f"{side}_center_y"] = []
            data[f"{side}_bbox_area"] = []
        
        # Process each frame
        for frame_filename, df_frame in raw_df.groupby("frame_filename"):
            if len(df_frame) < 2:
                continue
            
            # Get top 2 instances by keypoint score
            target_rows = df_frame.nlargest(2, "min_keypoint_score")
            
            # Skip if players overlap too much
            if not self._validate_player_positions(target_rows):
                continue
            
            # Determine left/right players by x-coordinate
            if target_rows.iloc[0]["center_x"] < target_rows.iloc[1]["center_x"]:
                left_row, right_row = target_rows.iloc[0], target_rows.iloc[1]
            else:
                left_row, right_row = target_rows.iloc[1], target_rows.iloc[0]
            
            # Add frame data
            self._add_frame_data(data, df_frame, left_row, right_row, frame_filename)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Normalize keypoints relative to bbox
        df = self._normalize_keypoints(df)
        
        return df
    
    def _validate_player_positions(self, target_rows: pd.DataFrame) -> bool:
        """Check if player positions are valid"""
        # Check if one player's center is inside the other's bbox
        return not (
            target_rows["center_y"].iloc[0] > target_rows["bbox_y_2"].iloc[1] or
            target_rows["center_y"].iloc[0] < target_rows["bbox_y_1"].iloc[1] or
            target_rows["center_y"].iloc[1] > target_rows["bbox_y_2"].iloc[0] or
            target_rows["center_y"].iloc[1] < target_rows["bbox_y_1"].iloc[0]
        )
    
    def _add_frame_data(
        self,
        data: Dict,
        df_frame: pd.DataFrame,
        left_row: pd.Series,
        right_row: pd.Series,
        frame_filename: str
    ):
        """Add data for a single frame"""
        # Basic info
        data["frame_filename"].append(frame_filename)
        data["video_filename"].append(df_frame["video_filename"].iloc[0])
        data["frame_idx"].append(df_frame["frame_idx"].iloc[0])
        
        # Action IDs
        left_action = df_frame["left_action"].iloc[0]
        right_action = df_frame["right_action"].iloc[0]
        data["left_action_id"].append(
            self.action_to_id.get(left_action, 0) if pd.notna(left_action) else 0
        )
        data["right_action_id"].append(
            self.action_to_id.get(right_action, 0) if pd.notna(right_action) else 0
        )
        
        # Keypoints and centers
        for i in range(self.config.num_kp):
            data[f"left_keypoint_{i}_x"].append(left_row[f"keypoint_{i}_x"])
            data[f"left_keypoint_{i}_y"].append(left_row[f"keypoint_{i}_y"])
            data[f"right_keypoint_{i}_x"].append(right_row[f"keypoint_{i}_x"])
            data[f"right_keypoint_{i}_y"].append(right_row[f"keypoint_{i}_y"])
        
        # Centers and areas
        for side, row in [("left", left_row), ("right", right_row)]:
            data[f"{side}_center_x"].append(row["center_x"])
            data[f"{side}_center_y"].append(row["center_y"])
            data[f"{side}_bbox_area"].append(row["bbox_area"])
    
    def _normalize_keypoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize keypoints relative to bbox center and area"""
        for side in ["left", "right"]:
            for i in range(self.config.num_kp):
                df[f"{side}_keypoint_{i}_x"] = (
                    df[f"{side}_keypoint_{i}_x"] - df[f"{side}_center_x"]
                ) / np.sqrt(df[f"{side}_bbox_area"])
                df[f"{side}_keypoint_{i}_y"] = (
                    df[f"{side}_keypoint_{i}_y"] - df[f"{side}_center_y"]
                ) / np.sqrt(df[f"{side}_bbox_area"])
        
        # Add distance feature
        df["distance"] = (
            df["right_center_x"] - df["left_center_x"]
        ) / np.sqrt(df["right_bbox_area"] + df["left_bbox_area"])
        
        # Drop unnecessary columns
        drop_cols = [
            "left_center_x", "left_center_y", "right_center_x", "right_center_y",
            "left_bbox_area", "right_bbox_area"
        ]
        df = df.drop(columns=drop_cols)
        
        return df
    
    def _add_pose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional pose features"""
        for side in ["left", "right"]:
            # Polar coordinates for each keypoint
            for i in range(self.config.num_kp):
                x_col = f"{side}_keypoint_{i}_x"
                y_col = f"{side}_keypoint_{i}_y"
                df[f"{side}_keypoint_{i}_dist"] = np.sqrt(df[x_col]**2 + df[y_col]**2)
                df[f"{side}_keypoint_{i}_angle"] = np.arctan2(df[y_col], df[x_col])
            
            # Joint angles
            df = self._add_joint_angles(df, side)
        
        return df
    
    def _add_joint_angles(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """Add joint angle features"""
        # Define joint connections
        angles = [
            ("shoulder_elbow", 5, 7),  # left shoulder to elbow
            ("elbow_wrist", 7, 9),     # left elbow to wrist
            ("shoulder_hip", 5, 11),   # left shoulder to hip
            ("hip_knee", 11, 13),      # left hip to knee
            ("knee_ankle", 13, 15),    # left knee to ankle
        ]
        
        # Add angles for both left and right sides
        for name, start_idx, end_idx in angles:
            # Left side
            df[f"{side}_l_{name}_angle"] = np.arctan2(
                df[f"{side}_keypoint_{end_idx}_y"] - df[f"{side}_keypoint_{start_idx}_y"],
                df[f"{side}_keypoint_{end_idx}_x"] - df[f"{side}_keypoint_{start_idx}_x"]
            )
            
            # Right side (mirror indices)
            start_idx_r = start_idx + 1 if start_idx % 2 == 1 else start_idx + 1
            end_idx_r = end_idx + 1 if end_idx % 2 == 1 else end_idx + 1
            df[f"{side}_r_{name}_angle"] = np.arctan2(
                df[f"{side}_keypoint_{end_idx_r}_y"] - df[f"{side}_keypoint_{start_idx_r}_y"],
                df[f"{side}_keypoint_{end_idx_r}_x"] - df[f"{side}_keypoint_{start_idx_r}_x"]
            )
        
        return df
    
    def _switch_side(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create switched version of data (left/right swap)"""
        df_switched = df.copy()
        
        # Keypoint mapping for left/right swap
        kpt_map = {
            0: 0,   # nose
            1: 2, 2: 1,   # eyes
            3: 4, 4: 3,   # ears
            5: 6, 6: 5,   # shoulders
            7: 8, 8: 7,   # elbows
            9: 10, 10: 9,   # wrists
            11: 12, 12: 11,   # hips
            13: 14, 14: 13,   # knees
            15: 16, 16: 15,   # ankles
        }
        
        # Swap action IDs
        df_switched["left_action_id"] = df["right_action_id"]
        df_switched["right_action_id"] = df["left_action_id"]
        
        # Swap keypoints
        for i in range(self.config.num_kp):
            j = kpt_map[i]
            df_switched[f"left_keypoint_{i}_x"] = -df[f"right_keypoint_{j}_x"]
            df_switched[f"left_keypoint_{i}_y"] = df[f"right_keypoint_{j}_y"]
            df_switched[f"right_keypoint_{i}_x"] = -df[f"left_keypoint_{j}_x"]
            df_switched[f"right_keypoint_{i}_y"] = df[f"left_keypoint_{j}_y"]
        
        return df_switched
    
    def _compute_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std for each feature"""
        # Get feature data
        feature_data = self.df[self.feature_cols].values
        
        # Compute statistics
        mean = np.mean(feature_data, axis=0)
        std = np.std(feature_data, axis=0)
        
        # Avoid division by zero
        std[std < 1e-6] = 1.0
        
        return {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32)
        }
    
    def _create_temporal_windows(self) -> List[Dict]:
        """Create temporal windows for sequence modeling"""
        data = []
        
        for switched in [False, True]:
            for video_filename in tqdm(
                self.df["video_filename"].unique(),
                desc=f"Creating windows (switched={switched})"
            ):
                df_video = self.df[
                    (self.df["video_filename"] == video_filename) &
                    (self.df["switched"] == switched)
                ].sort_values("frame_idx")
                
                frame_idxs = set(df_video["frame_idx"].values)
                if not frame_idxs:
                    continue
                
                max_frame_idx = max(frame_idxs)
                
                for center_idx in range(max_frame_idx + 1):
                    # Create window indices
                    window_idxs = set(range(
                        center_idx - self.config.window_width,
                        center_idx + self.config.window_width + 1
                    ))
                    
                    # Check if all window frames exist
                    if not window_idxs.issubset(frame_idxs):
                        continue
                    
                    # Get window data
                    window_df = df_video[df_video["frame_idx"].isin(window_idxs)]
                    
                    # Create feature tensor
                    x = torch.tensor(
                        window_df[self.feature_cols].values,
                        dtype=torch.float32
                    )
                    
                    # Apply normalization
                    x = self._normalize_features(x)
                    
                    # Get labels for center frame
                    center_row = df_video[df_video["frame_idx"] == center_idx].iloc[0]
                    y_left = F.one_hot(
                        torch.tensor(center_row["left_action_id"]),
                        num_classes=self.config.num_classes
                    ).float()
                    y_right = F.one_hot(
                        torch.tensor(center_row["right_action_id"]),
                        num_classes=self.config.num_classes
                    ).float()
                    
                    data.append({
                        "video_filename": video_filename,
                        "frame_filename": center_row["frame_filename"],
                        "frame_idx": center_idx,
                        "switched": switched,
                        "x": x,
                        "y_left": y_left,
                        "y_right": y_right,
                    })
        
        return data
    
    def _balance_samples(self, data: List[Dict]) -> List[Dict]:
        """Balance positive and negative samples"""
        # Separate positive and negative samples
        data_positive = [
            item for item in data
            if item["y_left"].argmax().item() != 0 or item["y_right"].argmax().item() != 0
        ]
        data_negative = [
            item for item in data
            if item["y_left"].argmax().item() == 0 and item["y_right"].argmax().item() == 0
        ]
        
        # Sample negative to match positive
        if len(data_negative) > len(data_positive):
            data_negative = random.sample(data_negative, len(data_positive))
        
        return data_positive + data_negative
    
    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize features using pre-computed statistics"""
        mean = torch.tensor(self.normalization_stats["mean"], dtype=torch.float32)
        std = torch.tensor(self.normalization_stats["std"], dtype=torch.float32)
        
        # Normalize: (x - mean) / std
        x_normalized = (x - mean) / std
        
        return x_normalized
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation"""
        # Add small Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        # Temporal jitter (small shift in time)
        if random.random() < 0.3 and x.shape[0] > 1:
            shift = random.choice([-1, 1])
            if shift == 1:
                x = torch.cat([x[0:1], x[:-1]], dim=0)
            else:
                x = torch.cat([x[1:], x[-1:]], dim=0)
        
        return x