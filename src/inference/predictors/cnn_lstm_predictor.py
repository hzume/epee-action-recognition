"""CNN-LSTM predictor for exp01/exp02."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..base import BasePredictor


class CNNLSTMPredictor(BasePredictor):
    """Predictor for CNN-LSTM models (exp01/exp02)."""
    
    def _load_model(self):
        # TODO: Implement CNN-LSTM model loading
        raise NotImplementedError("CNN-LSTM predictor not yet implemented")
        
    def _preprocess_video(self, video_path: Path) -> Dict:
        # TODO: Implement video preprocessing for CNN-LSTM
        raise NotImplementedError("CNN-LSTM predictor not yet implemented")
        
    def _predict_batch(self, batch: Dict) -> np.ndarray:
        # TODO: Implement batch prediction
        raise NotImplementedError("CNN-LSTM predictor not yet implemented")
        
    def _format_results(self, video_path: Path, predictions: np.ndarray, data: Dict) -> pd.DataFrame:
        # TODO: Implement result formatting
        raise NotImplementedError("CNN-LSTM predictor not yet implemented")