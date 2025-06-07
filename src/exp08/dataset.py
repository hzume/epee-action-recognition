"""Dataset module for exp08 - 3D pose processing"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm
import scipy.ndimage as ndimage

from .utils import Config3D, compute_3d_angle, apply_3d_rotation, normalize_3d_pose


class ThreeDPoseDataset(Dataset):
    """3D Pose dataset for fencing action recognition"""
    
    def __init__(
        self,
        data_path: str,
        config: Config3D,
        mode: str = "train",
        sample_balanced: bool = False,
        augment: bool = False
    ):
        """
        Initialize 3D pose dataset
        
        Args:
            data_path: Path to processed 3D pose CSV file
            config: Configuration object
            mode: Dataset mode ('train', 'val', 'test')
            sample_balanced: Whether to balance positive/negative samples
            augment: Whether to apply data augmentation
        """
        self.config = config
        self.mode = mode
        self.augment = augment and mode == "train"
        self.action_to_id = config.metadata["action_to_id"]
        
        # Load data
        print(f"Loading {mode} data from {data_path}")
        self.df = pd.read_csv(data_path)
        
        # Identify feature columns
        self._identify_feature_columns()
        
        # Create temporal windows
        self.data = self._create_temporal_windows()
        
        # Balance samples if requested
        if sample_balanced and mode == "train":
            self.data = self._balance_samples(self.data)
        
        print(f"Loaded {len(self.data)} samples for {mode}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Get item by index"""
        item = self.data[idx]
        pose_sequence = item["pose_sequence"]  # (seq_len, 17, 3)
        features = item["features"]  # (seq_len, num_features)
        
        # Apply augmentation if enabled
        if self.augment:
            pose_sequence, features = self._augment_3d_data(pose_sequence, features)
        
        # Combine pose and features
        x = self._combine_pose_and_features(pose_sequence, features)
        
        # Return additional info
        info = {
            "video_filename": item["video_filename"],
            "frame_idx": item["frame_idx"]
        }
        
        return x, item["y_left"], item["y_right"], info
    
    def _identify_feature_columns(self):
        """Identify different types of feature columns"""
        all_cols = self.df.columns.tolist()
        
        # Metadata columns
        self.metadata_cols = [
            'frame_filename', 'video_filename', 'frame_idx', 'labels',
            'left_action', 'left_outcome', 'right_action', 'right_outcome',
            'instance_id', 'width', 'height'
        ]
        
        # 3D keypoint columns
        self.keypoint_3d_cols = []
        for i in range(17):
            for axis in ['x', 'y', 'z']:
                col = f"keypoint_{i}_{axis}"
                if col in all_cols:
                    self.keypoint_3d_cols.append(col)
        
        # Keypoint score columns
        self.score_cols = [f"keypoint_score_{i}" for i in range(17) if f"keypoint_score_{i}" in all_cols]
        
        # Additional feature columns
        self.feature_cols = [
            col for col in all_cols 
            if col not in self.metadata_cols + self.keypoint_3d_cols + self.score_cols
        ]
        
        print(f"Found {len(self.keypoint_3d_cols)} 3D keypoint coordinates")
        print(f"Found {len(self.score_cols)} keypoint scores") 
        print(f"Found {len(self.feature_cols)} additional features")
    
    def _create_temporal_windows(self) -> List[Dict]:
        """Create temporal windows for sequence modeling"""
        data = []
        window_width = self.config.data["window_width"]
        
        for video_filename in tqdm(self.df["video_filename"].unique(), desc=f"Creating {self.mode} windows"):
            df_video = self.df[self.df["video_filename"] == video_filename].sort_values("frame_idx")
            
            # Group by instance_id to handle multiple people per frame
            for instance_id in df_video["instance_id"].unique():
                df_instance = df_video[df_video["instance_id"] == instance_id]
                frame_idxs = set(df_instance["frame_idx"].values)
                
                if not frame_idxs:
                    continue
                
                max_frame_idx = max(frame_idxs)
                
                for center_idx in range(max_frame_idx + 1):
                    # Create window indices
                    window_idxs = set(range(
                        center_idx - window_width,
                        center_idx + window_width + 1
                    ))
                    
                    # Check if all window frames exist
                    if not window_idxs.issubset(frame_idxs):
                        continue
                    
                    # Get window data
                    window_df = df_instance[df_instance["frame_idx"].isin(window_idxs)].sort_values("frame_idx")
                    
                    if len(window_df) != len(window_idxs):
                        continue
                    
                    # Extract 3D pose sequence
                    pose_sequence = self._extract_pose_sequence(window_df)
                    
                    # Extract additional features
                    features = self._extract_features(window_df)
                    
                    # Get labels for center frame
                    center_row = df_instance[df_instance["frame_idx"] == center_idx].iloc[0]
                    
                    # Process action labels
                    left_action = center_row["left_action"]
                    right_action = center_row["right_action"]
                    
                    left_action_id = self.action_to_id.get(left_action, 0) if pd.notna(left_action) else 0
                    right_action_id = self.action_to_id.get(right_action, 0) if pd.notna(right_action) else 0
                    
                    y_left = F.one_hot(
                        torch.tensor(left_action_id),
                        num_classes=self.config.num_classes
                    ).float()
                    y_right = F.one_hot(
                        torch.tensor(right_action_id),
                        num_classes=self.config.num_classes
                    ).float()
                    
                    data.append({
                        "video_filename": video_filename,
                        "frame_idx": center_idx,
                        "instance_id": instance_id,
                        "pose_sequence": pose_sequence,
                        "features": features,
                        "y_left": y_left,
                        "y_right": y_right,
                    })
        
        return data
    
    def _extract_pose_sequence(self, window_df: pd.DataFrame) -> torch.Tensor:
        """Extract 3D pose sequence from window dataframe"""
        poses = []
        
        for _, row in window_df.iterrows():
            # Extract 3D keypoints
            pose_3d = np.zeros((17, 3))
            for i in range(17):
                pose_3d[i, 0] = row[f"keypoint_{i}_x"]
                pose_3d[i, 1] = row[f"keypoint_{i}_y"] 
                pose_3d[i, 2] = row[f"keypoint_{i}_z"]
            
            # Normalize pose if configured
            if self.config.data["normalize_poses"]:
                if self.config.data["center_on_hip"]:  # Actually center on pelvis for Human3D
                    pose_3d = normalize_3d_pose(pose_3d)
            
            poses.append(pose_3d)
        
        return torch.tensor(np.array(poses), dtype=torch.float32)  # (seq_len, 17, 3)
    
    def _extract_features(self, window_df: pd.DataFrame) -> torch.Tensor:
        """Extract additional features from window dataframe"""
        if not self.feature_cols:
            # Return empty features if no additional features
            return torch.zeros(len(window_df), 0, dtype=torch.float32)
        
        features = window_df[self.feature_cols].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        
        return torch.tensor(features, dtype=torch.float32)  # (seq_len, num_features)
    
    def _combine_pose_and_features(self, pose_sequence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Combine 3D pose sequence with additional features"""
        seq_len = pose_sequence.shape[0]
        
        # Flatten pose sequence: (seq_len, 17, 3) -> (seq_len, 51)
        pose_flat = pose_sequence.view(seq_len, -1)
        
        if features.shape[1] > 0:
            # Concatenate with additional features
            combined = torch.cat([pose_flat, features], dim=1)
        else:
            combined = pose_flat
        
        return combined
    
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
        
        print(f"Before balancing: {len(data_positive)} positive, {len(data_negative)} negative")
        
        # Sample negative to match positive
        if len(data_negative) > len(data_positive):
            data_negative = random.sample(data_negative, len(data_positive))
        
        balanced_data = data_positive + data_negative
        print(f"After balancing: {len(balanced_data)} total samples")
        
        return balanced_data
    
    def _augment_3d_data(self, pose_sequence: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation to 3D pose data"""
        aug_config = self.config.training["augmentation"]
        
        # Gaussian noise
        if random.random() < 0.5:
            noise_std = aug_config["gaussian_noise_std"]
            noise = torch.randn_like(pose_sequence) * noise_std
            pose_sequence = pose_sequence + noise
        
        # 3D rotation
        if random.random() < 0.3:
            rotation_range = aug_config["rotation_range"]
            angles = np.random.uniform(
                -np.radians(rotation_range), 
                np.radians(rotation_range), 
                size=3
            )
            
            # Apply rotation to each frame
            for t in range(pose_sequence.shape[0]):
                pose_np = pose_sequence[t].numpy()
                pose_rotated = apply_3d_rotation(pose_np, angles)
                pose_sequence[t] = torch.from_numpy(pose_rotated)
        
        # Scale variation
        if random.random() < 0.3:
            scale_range = aug_config["scale_range"]
            scale = 1.0 + random.uniform(-scale_range, scale_range)
            pose_sequence = pose_sequence * scale
        
        # Temporal jitter
        if random.random() < 0.2 and aug_config["temporal_jitter"] and pose_sequence.shape[0] > 1:
            # Small temporal shift
            shift = random.choice([-1, 1])
            if shift == 1:
                pose_sequence = torch.cat([pose_sequence[0:1], pose_sequence[:-1]], dim=0)
                if features.shape[1] > 0:
                    features = torch.cat([features[0:1], features[:-1]], dim=0)
            else:
                pose_sequence = torch.cat([pose_sequence[1:], pose_sequence[-1:]], dim=0)
                if features.shape[1] > 0:
                    features = torch.cat([features[1:], features[-1:]], dim=0)
        
        return pose_sequence, features


def create_3d_pose_datasets(config: Config3D) -> Tuple[ThreeDPoseDataset, ThreeDPoseDataset, ThreeDPoseDataset]:
    """Create train, validation, and test datasets"""
    data_dir = config.data_dir
    
    train_dataset = ThreeDPoseDataset(
        data_path=data_dir / "train_3d.csv",
        config=config,
        mode="train",
        sample_balanced=True,
        augment=True
    )
    
    val_dataset = ThreeDPoseDataset(
        data_path=data_dir / "val_3d.csv",
        config=config,
        mode="val",
        sample_balanced=False,
        augment=False
    )
    
    test_dataset = ThreeDPoseDataset(
        data_path=data_dir / "test_3d.csv",
        config=config,
        mode="test",
        sample_balanced=False,
        augment=False
    )
    
    return train_dataset, val_dataset, test_dataset