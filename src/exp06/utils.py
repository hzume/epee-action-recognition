"""Utility functions for exp05"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix


class Config:
    """Configuration class for experiment"""
    
    def __init__(self, config_path: str = None):
        # Default configuration
        self.data_dir = Path("input/data_10hz")
        self.output_dir = Path("outputs/exp06")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        with open(self.data_dir.parent / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        # Model parameters
        self.num_classes = len(self.metadata["action_to_id"]) + 1
        self.num_kp = 17
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 1e-2
        self.weight_decay = 1e-4
        
        # Data parameters
        self.window_width = 2
        self.min_keypoint_score = 0.2
        self.min_bbox_ratio = 0.0375
        self.max_bbox_ratio = 0.15
        
        # Cross-validation
        self.n_folds = 4
        self.fold = 0  # Which fold to use for validation
        self.seed = 42
        
        # Prediction videos
        self.predict_videos = [
            '2024-11-10-18-33-49.mp4',
            '2024-11-10-19-21-45.mp4',
            '2025-01-04_08-37-18.mp4',
            '2025-01-04_08-40-12.mp4'
        ]
        
        # Model architecture
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.1
        self.bidirectional = True
        
        # Temporal consistency
        self.kl_weight = 0.1  # Weight for KL divergence loss
        self.temperature = 2.0  # Temperature for softmax in KL computation
        
        # Override with config file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Evaluate predictions and return metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        prefix: Prefix for metric names
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Overall accuracy
    metrics[f"{prefix}accuracy"] = (y_true == y_pred).mean()
    
    # Accuracy ignoring 'none' class (class 0)
    mask_not_none = y_true != 0
    if mask_not_none.sum() > 0:
        metrics[f"{prefix}accuracy_ignore_none"] = (
            y_true[mask_not_none] == y_pred[mask_not_none]
        ).mean()
    else:
        metrics[f"{prefix}accuracy_ignore_none"] = 0.0
    
    # Get unique labels that actually appear in the data
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    # F1 scores
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


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Classification Report"
):
    """Print detailed classification report"""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Get unique labels that actually appear in the data
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Filter class names to match the labels
    filtered_class_names = [class_names[i] for i in labels if i < len(class_names)]
    
    print(classification_report(
        y_true, 
        y_pred, 
        labels=labels,
        target_names=filtered_class_names, 
        zero_division=0
    ))


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    config: Config
):
    """Save predictions to CSV file"""
    # Map action IDs to names
    id_to_action = {v: k for k, v in config.metadata["action_to_id"].items()}
    id_to_action[0] = "none"
    
    # Add action names
    for side in ["left", "right"]:
        if f"{side}_pred_action_id" in predictions.columns:
            predictions[f"{side}_pred_action"] = predictions[f"{side}_pred_action_id"].map(id_to_action)
    
    # Save to CSV
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def save_normalization_stats(stats: Dict[str, np.ndarray], path: Path):
    """Save normalization statistics to file"""
    np_stats = {
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist()
    }
    with open(path, 'w') as f:
        json.dump(np_stats, f, indent=2)
    print(f"Normalization stats saved to {path}")


def load_normalization_stats(path: Path) -> Dict[str, np.ndarray]:
    """Load normalization statistics from file"""
    with open(path, 'r') as f:
        np_stats = json.load(f)
    
    stats = {
        "mean": np.array(np_stats["mean"], dtype=np.float32),
        "std": np.array(np_stats["std"], dtype=np.float32)
    }
    return stats