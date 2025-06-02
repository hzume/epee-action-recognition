"""LSTM predictor for exp04 (bone features)."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.inference.base import BasePredictor


class LSTMModel(nn.Module):
    """LSTM model for bone feature sequences."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.5, bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output
        if self.bidirectional:
            # Concatenate forward and backward outputs from last timestep
            out = lstm_out[:, -1, :]
        else:
            out = lstm_out[:, -1, :]
            
        out = self.dropout(out)
        out = self.fc(out)
        return out


class BoneFeatureDataset(Dataset):
    """Dataset for bone features."""
    
    def __init__(self, features: np.ndarray, window_size: int):
        self.features = features
        self.window_size = window_size
        
    def __len__(self):
        return len(self.features) - self.window_size + 1
    
    def __getitem__(self, idx):
        window = self.features[idx:idx + self.window_size]
        return torch.FloatTensor(window)


class LSTMPredictor(BasePredictor):
    """Predictor for LSTM models using bone features."""
    
    def _load_model(self) -> nn.Module:
        """Load LSTM model from checkpoint."""
        # Calculate input size based on bones and features
        n_bones = len(self.config.data.bones)
        n_features = 2  # length and angle for each bone
        input_size = n_bones * n_features
        
        # Create model
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.model.hidden_size,
            num_layers=self.config.model.num_layers,
            num_classes=self.config.model.num_classes,
            dropout=self.config.model.dropout,
            bidirectional=self.config.model.bidirectional
        )
        
        # Load checkpoint
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")
            
        model.to(self.device)
        model.eval()
        return model
    
    def _extract_bone_features(self, pose_data: pd.DataFrame) -> np.ndarray:
        """Extract bone features from pose keypoints."""
        bones = self.config.data.bones
        n_frames = len(pose_data)
        n_bones = len(bones)
        features = np.zeros((n_frames, n_bones * 2))  # length and angle
        
        for frame_idx in range(n_frames):
            frame = pose_data.iloc[frame_idx]
            
            for bone_idx, (start_idx, end_idx) in enumerate(bones):
                # Get keypoint coordinates
                start_x = frame[f"x{start_idx}"]
                start_y = frame[f"y{start_idx}"]
                end_x = frame[f"x{end_idx}"]
                end_y = frame[f"y{end_idx}"]
                
                # Calculate bone length
                length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                
                # Calculate bone angle
                angle = np.arctan2(end_y - start_y, end_x - start_x)
                
                # Store features
                features[frame_idx, bone_idx * 2] = length
                features[frame_idx, bone_idx * 2 + 1] = angle
                
        # Normalize features
        if self.config.preprocessing.normalize_pose:
            # Normalize lengths by average bone length
            avg_length = np.mean(features[:, ::2])  # Every other column is length
            if avg_length > 0:
                features[:, ::2] /= avg_length
                
        return features
    
    def _preprocess_video(self, video_path: Path) -> Dict:
        """Preprocess video by extracting pose and bone features."""
        # Load pose predictions
        data_dir = Path(self.config.data.data_dir)
        pose_file = data_dir / self.config.data.pose_file
        
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose predictions not found: {pose_file}")
            
        pose_df = pd.read_csv(pose_file)
        
        # Filter for this video
        video_name = video_path.stem
        video_poses = pose_df[pose_df["video_filename"] == f"{video_name}.mp4"]
        
        if len(video_poses) == 0:
            raise ValueError(f"No pose data found for video: {video_name}")
            
        # Sort by frame index
        video_poses = video_poses.sort_values("frame_idx").reset_index(drop=True)
        
        # Extract bone features
        features = self._extract_bone_features(video_poses)
        
        return {
            "features": features,
            "video_name": video_name,
            "frame_indices": video_poses["frame_idx"].values
        }
    
    def _predict_batch(self, data: Dict) -> np.ndarray:
        """Perform batch prediction on preprocessed data."""
        features = data["features"]
        window_size = self.config.data.window_size
        
        # Create dataset and dataloader
        dataset = BoneFeatureDataset(features, window_size)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.inference.batch_size,
            shuffle=False
        )
        
        # Perform inference
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                
        # Pad predictions to match original length
        # (predictions are shorter due to windowing)
        predictions = np.array(predictions)
        padding = window_size - 1
        predictions = np.pad(predictions, (padding, 0), constant_values=5)  # 5 = "none"
        
        return predictions
    
    def _format_results(self, video_path: Path, predictions: np.ndarray, data: Dict) -> pd.DataFrame:
        """Format predictions as DataFrame."""
        video_name = data["video_name"]
        frame_indices = data["frame_indices"]
        
        # Create results DataFrame
        results = pd.DataFrame({
            "video_filename": f"{video_name}.mp4",
            "frame_idx": frame_indices,
            "pred_action_id": predictions,
            "pred_action": [self.id_to_action[i] for i in predictions]
        })
        
        # Add probability columns for compatibility
        for action in self.action_names:
            results[f"prob_{action}"] = 0.0
            
        # Set probability 1.0 for predicted action
        for idx, pred_id in enumerate(predictions):
            action = self.id_to_action[pred_id]
            results.loc[idx, f"prob_{action}"] = 1.0
            
        return results