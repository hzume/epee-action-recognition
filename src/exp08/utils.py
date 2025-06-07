"""Utility functions for exp08 - 3D pose processing"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import torch


class Config3D:
    """Configuration class for 3D pose experiment"""
    
    def __init__(self, config_path: str = None):
        self.debug = False

        # Default configuration
        self.data_dir = Path("input/data_10hz_3d")
        self.output_dir = Path("outputs/exp08")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_path = Path("input/metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            # Default metadata
            self.metadata = {
                "action_to_id": {
                    "lunge": 1,
                    "fleche": 2, 
                    "counter": 3,
                    "parry": 4,
                    "prime": 5
                }
            }
        
        # Model parameters
        self.num_classes = len(self.metadata["action_to_id"]) + 1  # +1 for 'none'
        self.num_keypoints = 17  # Human3D format
        
        # Default model configuration
        self.model = {
            "type": "3d_pose_lstm",
            "hidden_size": 512,
            "num_layers": 3,
            "dropout": 0.2,
            "bidirectional": True,
            "use_spatial_attention": True,
            "use_joint_embeddings": True,
            "joint_embedding_dim": 64,
            "spatial_attention_heads": 8,
            "use_3d_features": True,
            "use_velocity_features": True,
            "use_angle_features": True,
            "use_distance_features": True
        }
        
        # Training parameters
        self.training = {
            "batch_size": 16,
            "epochs": 100,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_steps": 0.1,
            "augmentation": {
                "gaussian_noise_std": 0.01,
                "rotation_range": 10,
                "scale_range": 0.1,
                "temporal_jitter": True
            }
        }
        
        # Data processing
        self.data = {
            "window_width": 5,
            "min_keypoint_score": 0.3,
            "normalize_poses": True,
            "center_on_hip": True,
            "remove_depth_ambiguity": True,
            "smooth_trajectories": True,
            "filter_outliers": True
        }
        
        # Cross-validation
        self.cross_validation = {
            "n_folds": 4,
            "seed": 42,
            "stratify_by": "left_action"
        }
        
        # Prediction videos
        self.predict_videos = [
            "2024-11-10-18-33-49.mp4",
            "2024-11-10-19-21-45.mp4",
            "2025-01-04_08-37-18.mp4",
            "2025-01-04_08-40-12.mp4"
        ]
        
        # Loss configuration
        self.loss = {
            "type": "weighted_cross_entropy",
            "class_weights": {
                "none": 0.1,
                "lunge": 1.0,
                "fleche": 1.0,
                "counter": 1.0,
                "parry": 1.0,
                "prime": 1.0
            },
            "temporal_consistency_weight": 0.1,
            "spatial_consistency_weight": 0.05
        }
        
        # Load config file if provided
        if config_path and Path(config_path).exists():
            self._load_config_file(config_path)
        
        # Ensure path objects
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config_file(self, config_path: str):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        # Update configuration recursively
        def update_nested_dict(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested_dict(target[key], value)
                else:
                    target[key] = value
        
        # Update each section
        for section in ['model', 'training', 'data', 'cross_validation', 'loss']:
            if section in config_data:
                if hasattr(self, section):
                    update_nested_dict(getattr(self, section), config_data[section])
                else:
                    setattr(self, section, config_data[section])
        
        # Update direct attributes
        for key, value in config_data.items():
            if key not in ['model', 'training', 'data', 'cross_validation', 'loss']:
                setattr(self, key, value)
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'num_classes': self.num_classes,
            'num_keypoints': self.num_keypoints,
            'model': self.model,
            'training': self.training,
            'data': self.data,
            'cross_validation': self.cross_validation,
            'predict_videos': self.predict_videos,
            'loss': self.loss,
            'metadata': self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


def compute_3d_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute angle between three 3D points with p2 as vertex"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Handle zero vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm == 0 or v2_norm == 0:
        return 0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    # Clip to handle numerical errors
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return angle


def apply_3d_rotation(points: np.ndarray, angles: Tuple[float, float, float]) -> np.ndarray:
    """Apply 3D rotation to points
    
    Args:
        points: Array of shape (N, 3) with 3D points
        angles: Rotation angles in radians (roll, pitch, yaw)
    
    Returns:
        Rotated points
    """
    roll, pitch, yaw = angles
    
    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = R_z @ R_y @ R_x
    
    # Apply rotation
    return points @ R.T


def normalize_3d_pose(pose: np.ndarray, pelvis_index: int = 0) -> np.ndarray:
    """Normalize 3D pose by centering on pelvis and scaling
    
    Args:
        pose: Array of shape (17, 3) with 3D keypoints in Human3D format
        pelvis_index: Index of pelvis keypoint (0 in Human3D)
    
    Returns:
        Normalized pose
    """
    # Center on pelvis (should already be centered in Human3D format)
    pelvis_center = pose[pelvis_index]
    pose_centered = pose - pelvis_center
    
    # Scale by torso length (pelvis to thorax)
    thorax_index = 8  # thorax in Human3D format
    torso_length = np.linalg.norm(pose[thorax_index] - pose[pelvis_index])
    
    if torso_length > 0:
        pose_normalized = pose_centered / torso_length
    else:
        pose_normalized = pose_centered
    
    return pose_normalized


def evaluate_3d_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    prefix: str = ""
) -> Dict[str, float]:
    """Evaluate 3D pose predictions with additional metrics"""
    metrics = {}
    
    # Standard metrics
    metrics[f"{prefix}accuracy"] = (y_true == y_pred).mean()
    
    # Accuracy ignoring 'none' class
    mask_not_none = y_true != 0
    if mask_not_none.sum() > 0:
        metrics[f"{prefix}accuracy_ignore_none"] = (
            y_true[mask_not_none] == y_pred[mask_not_none]
        ).mean()
    else:
        metrics[f"{prefix}accuracy_ignore_none"] = 0.0
    
    # F1 scores
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    metrics[f"{prefix}f1_macro"] = f1_score(
        y_true, y_pred, labels=unique_labels, average='macro', zero_division=0
    )
    metrics[f"{prefix}f1_weighted"] = f1_score(
        y_true, y_pred, labels=unique_labels, average='weighted', zero_division=0
    )
    
    # Per-class F1 scores
    f1_scores = f1_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
    for i, label in enumerate(unique_labels):
        if label < len(class_names):
            class_name = class_names[label]
            metrics[f"{prefix}f1_{class_name}"] = f1_scores[i]
    
    return metrics


def print_3d_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "3D Pose Classification Report"
):
    """Print detailed classification report for 3D poses"""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Get unique labels
    labels = sorted(list(set(y_true) | set(y_pred)))
    filtered_class_names = [class_names[i] for i in labels if i < len(class_names)]
    
    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels,
        target_names=filtered_class_names, 
        zero_division=0
    ))


def save_3d_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    config: Config3D
):
    """Save 3D pose predictions to CSV"""
    # Map action IDs to names
    id_to_action = {v: k for k, v in config.metadata["action_to_id"].items()}
    id_to_action[0] = "none"
    
    # Add action names
    for side in ["left", "right"]:
        if f"{side}_pred_action_id" in predictions.columns:
            predictions[f"{side}_pred_action"] = predictions[f"{side}_pred_action_id"].map(id_to_action)
    
    # Save to CSV
    predictions.to_csv(output_path, index=False)
    print(f"3D pose predictions saved to {output_path}")


def load_3d_pose_data(data_path: Path) -> pd.DataFrame:
    """Load processed 3D pose data"""
    if not data_path.exists():
        raise FileNotFoundError(f"3D pose data not found at {data_path}")
    
    return pd.read_csv(data_path)


def create_joint_connectivity_matrix() -> torch.Tensor:
    """Create connectivity matrix for Human3D skeleton"""
    # Human3D keypoint connections
    connections = [
        # Lower body connections
        (0, 1), (0, 4),  # pelvis to hips
        (1, 2), (2, 3),  # left hip to knee to ankle
        (4, 5), (5, 6),  # right hip to knee to ankle
        # Spine connections
        (0, 7), (7, 8), (8, 9), (9, 10),  # pelvis to spine to thorax to neck to head
        # Upper body connections
        (8, 11), (8, 14),  # thorax to shoulders
        (11, 12), (12, 13),  # left shoulder to elbow to wrist
        (14, 15), (15, 16),  # right shoulder to elbow to wrist
    ]
    
    # Create adjacency matrix
    adj_matrix = torch.zeros(17, 17)
    for i, j in connections:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1
    
    # Add self-connections
    adj_matrix += torch.eye(17)
    
    return adj_matrix