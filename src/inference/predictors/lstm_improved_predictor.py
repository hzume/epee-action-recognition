"""LSTM predictor for improved exp05 model."""

from pathlib import Path
from typing import Dict, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..base import BasePredictor
from ...exp05.model import ImprovedEpeeModel
from ...exp05.dataset import ImprovedEpeeDataset, CFG


class InferenceLSTMDataset(Dataset):
    """Dataset for inference without labels."""
    
    def __init__(self, df: pd.DataFrame, feature_cols: list):
        self.df = df.sort_values(by="frame_idx")
        self.feature_cols = feature_cols
        self.window_width = CFG.window_width
        
        # Build valid frame indices
        self.valid_indices = []
        frame_idxs = set(self.df["frame_idx"].values)
        last_frame_idx = max(frame_idxs)
        
        for i in range(last_frame_idx):
            window = set(range(i - self.window_width, i + self.window_width + 1))
            if window.issubset(frame_idxs):
                self.valid_indices.append(i)
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        frame_idx = self.valid_indices[idx]
        window = range(frame_idx - self.window_width, frame_idx + self.window_width + 1)
        
        x = torch.tensor(
            self.df[self.df["frame_idx"].isin(window)][self.feature_cols].values
        ).float()
        
        return x, frame_idx


class ImprovedLSTMPredictor(BasePredictor):
    """Predictor for improved LSTM model (exp05)."""
    
    def _load_model(self) -> torch.nn.Module:
        """Load the improved LSTM model."""
        # Get model configuration
        model_config = self.config.model
        
        # First, we need to get the number of features
        # This is a bit tricky since we need to process dummy data
        dummy_df = pd.DataFrame({
            'frame_filename': ['dummy.jpg'],
            'video_filename': ['dummy.mp4'],
            'frame_idx': [0],
            'left_action': ['none'],
            'right_action': ['none'],
            'height': [720],
            'width': [1280],
        })
        
        # Add dummy pose data
        for side in ['left', 'right']:
            for i in range(17):
                dummy_df[f'keypoint_{i}_x'] = [100]
                dummy_df[f'keypoint_{i}_y'] = [100]
                dummy_df[f'keypoint_score_{i}'] = [0.9]
            dummy_df[f'bbox_x_1'] = [50]
            dummy_df[f'bbox_y_1'] = [50]
            dummy_df[f'bbox_x_2'] = [150]
            dummy_df[f'bbox_y_2'] = [250]
        
        # Create temporary dataset to get feature columns
        temp_ds = ImprovedEpeeDataset(dummy_df, mode="test", balance_samples=False)
        num_features = len(temp_ds.feature_cols)
        
        # Create model
        model = ImprovedEpeeModel(
            num_features=num_features,
            num_classes=CFG.num_classes,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            num_heads=model_config.get('num_heads', 4),
            dropout=0,  # No dropout during inference
            use_attention=model_config.get('use_attention', True)
        )
        
        # Load checkpoint
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                # Remove 'model.' prefix if present
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _preprocess_video(self, video_path: Path) -> Dict:
        """Preprocess video for LSTM inference."""
        # Load pose predictions
        data_dir = self.config.get('data_dir', Path("input/data_10hz"))
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        pose_df = pd.read_csv(data_dir / "pose_preds.csv")
        
        # Filter for this video
        video_name = video_path.name
        video_df = pose_df[pose_df["video_filename"] == video_name]
        
        if len(video_df) == 0:
            raise ValueError(f"No pose predictions found for video: {video_name}")
        
        # Create dataset for preprocessing
        dataset = ImprovedEpeeDataset(video_df, mode="test", balance_samples=False)
        
        return {
            'dataset': dataset,
            'video_df': video_df,
            'video_name': video_name
        }
    
    def _predict_batch(self, batch: Dict) -> np.ndarray:
        """Perform batch prediction."""
        dataset = batch['dataset']
        video_df = batch['video_df']
        video_name = batch['video_name']
        
        # Process both original and switched versions
        all_predictions = []
        
        for switched in [False, True]:
            df_filtered = dataset.df[
                (dataset.df["video_filename"] == video_name) & 
                (dataset.df["switched"] == switched)
            ]
            
            if len(df_filtered) == 0:
                continue
            
            # Create inference dataset
            inference_ds = InferenceLSTMDataset(df_filtered, dataset.feature_cols)
            
            if len(inference_ds) == 0:
                continue
            
            # Create dataloader
            dataloader = DataLoader(
                inference_ds,
                batch_size=self.config.inference.get('batch_size', 64),
                shuffle=False,
                num_workers=0
            )
            
            # Predictions for this version
            frame_predictions = {}
            
            with torch.no_grad():
                for x_batch, frame_idx_batch in tqdm(dataloader, desc=f"Predicting (switched={switched})"):
                    x_batch = x_batch.to(self.device)
                    
                    # Get predictions
                    left_logits, right_logits = self.model(x_batch)
                    
                    # Convert to probabilities
                    left_probs = torch.softmax(left_logits, dim=-1)
                    right_probs = torch.softmax(right_logits, dim=-1)
                    
                    # Store predictions
                    for i, frame_idx in enumerate(frame_idx_batch):
                        frame_idx = int(frame_idx)
                        if switched:
                            # Swap predictions for switched version
                            frame_predictions[frame_idx] = {
                                'left': right_probs[i].cpu().numpy(),
                                'right': left_probs[i].cpu().numpy()
                            }
                        else:
                            frame_predictions[frame_idx] = {
                                'left': left_probs[i].cpu().numpy(),
                                'right': right_probs[i].cpu().numpy()
                            }
            
            all_predictions.append(frame_predictions)
        
        # Merge predictions (average if both versions available)
        merged_predictions = {}
        all_frame_idxs = set()
        for preds in all_predictions:
            all_frame_idxs.update(preds.keys())
        
        for frame_idx in sorted(all_frame_idxs):
            probs_left = []
            probs_right = []
            
            for preds in all_predictions:
                if frame_idx in preds:
                    probs_left.append(preds[frame_idx]['left'])
                    probs_right.append(preds[frame_idx]['right'])
            
            if probs_left:
                merged_predictions[frame_idx] = {
                    'left': np.mean(probs_left, axis=0),
                    'right': np.mean(probs_right, axis=0)
                }
        
        # Convert to array format
        num_frames = len(video_df["frame_idx"].unique())
        predictions = np.zeros((num_frames, 2), dtype=np.int32)
        
        for i, frame_idx in enumerate(sorted(video_df["frame_idx"].unique())):
            if frame_idx in merged_predictions:
                predictions[i, 0] = merged_predictions[frame_idx]['left'].argmax()
                predictions[i, 1] = merged_predictions[frame_idx]['right'].argmax()
            else:
                # Default to "none" for frames without predictions
                predictions[i, 0] = 0
                predictions[i, 1] = 0
        
        return predictions
    
    def _format_results(self, video_path: Path, predictions: np.ndarray, data: Dict) -> pd.DataFrame:
        """Format predictions as DataFrame."""
        video_df = data['video_df']
        
        # Get unique frames
        unique_frames = video_df.drop_duplicates('frame_idx').sort_values('frame_idx')
        
        results = []
        for i, (_, row) in enumerate(unique_frames.iterrows()):
            if i < len(predictions):
                left_pred = self.id_to_action[predictions[i, 0]]
                right_pred = self.id_to_action[predictions[i, 1]]
            else:
                left_pred = "none"
                right_pred = "none"
            
            results.append({
                'video_filename': video_path.name,
                'frame_idx': row['frame_idx'],
                'left_pred_action': left_pred,
                'right_pred_action': right_pred
            })
        
        return pd.DataFrame(results)