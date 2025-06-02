"""Base predictor class for unified inference interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf


class BasePredictor(ABC):
    """Base class for all predictors."""

    def __init__(self, config_path: Union[str, Path], checkpoint_path: Optional[Union[str, Path]] = None):
        """Initialize predictor with configuration and model checkpoint.
        
        Args:
            config_path: Path to configuration file (YAML)
            checkpoint_path: Path to model checkpoint. If None, uses path from config.
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Use checkpoint from config if not provided
        if checkpoint_path is None:
            checkpoint_path = self.config.inference.checkpoint
        self.checkpoint_path = Path(checkpoint_path)
        
        # Initialize model
        self.device = torch.device(self.config.inference.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self._load_model()
        
        # Action mapping
        self.action_names = self.config.get("action_names", 
            ["lunge", "fleche", "counter", "parry", "prime", "none"])
        self.action_to_id = {name: i for i, name in enumerate(self.action_names)}
        self.id_to_action = {i: name for i, name in enumerate(self.action_names)}
        
    def _load_config(self) -> DictConfig:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        return OmegaConf.load(self.config_path)
    
    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _preprocess_video(self, video_path: Path) -> Dict:
        """Preprocess video for inference. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict_batch(self, batch: Dict) -> np.ndarray:
        """Perform batch prediction. Must be implemented by subclasses."""
        pass
    
    def predict_video(self, video_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None,
                      output_video_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Predict actions for entire video.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save predictions as CSV
            output_video_path: Optional path to save labeled video
            
        Returns:
            DataFrame with predictions
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Preprocess video
        data = self._preprocess_video(video_path)
        
        # Perform prediction
        predictions = self._predict_batch(data)
        
        # Format results
        results = self._format_results(video_path, predictions, data)
        
        # Save CSV if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            
        # Save video if requested
        if output_video_path is not None:
            from .utils import create_labeled_video
            create_labeled_video(
                video_path=video_path,
                predictions_df=results,
                output_path=output_video_path,
                fps=self.config.get("fps", 10)
            )
            
        return results
    
    def predict_batch_videos(self, video_paths: List[Union[str, Path]], 
                            output_dir: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
        """Predict actions for multiple videos.
        
        Args:
            video_paths: List of video paths
            output_dir: Optional directory to save predictions
            
        Returns:
            Dictionary mapping video names to prediction DataFrames
        """
        results = {}
        
        for video_path in video_paths:
            video_path = Path(video_path)
            video_name = video_path.stem
            
            # Determine output path
            output_path = None
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{video_name}_predictions.csv"
            
            # Predict
            try:
                df = self.predict_video(video_path, output_path)
                results[video_name] = df
                print(f"✓ Processed: {video_name}")
            except Exception as e:
                print(f"✗ Failed: {video_name} - {str(e)}")
                results[video_name] = None
                
        return results
    
    @abstractmethod
    def _format_results(self, video_path: Path, predictions: np.ndarray, data: Dict) -> pd.DataFrame:
        """Format predictions as DataFrame. Must be implemented by subclasses."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config_path.name}, checkpoint={self.checkpoint_path.name})"