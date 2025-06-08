"""Simple API for loading predictors."""

from pathlib import Path
from typing import Optional, Union

from .base import BasePredictor
from .predictors import (
    CNNPredictor,
    CNNLSTMPredictor,
    LightGBMPredictor,
    LSTMPredictor,
)

# Import improved predictor if available
try:
    from .predictors.lstm_improved_predictor import ImprovedLSTMPredictor
except ImportError:
    ImprovedLSTMPredictor = None


PREDICTOR_REGISTRY = {
    "exp00": CNNPredictor,
    "exp00_cnn": CNNPredictor,
    "exp01": CNNLSTMPredictor,
    "exp01_cnn_lstm": CNNLSTMPredictor,
    "exp02": CNNLSTMPredictor,
    "exp02_cnn_lstm": CNNLSTMPredictor,
    "exp03": LightGBMPredictor,
    "exp03_lightgbm": LightGBMPredictor,
    "exp04": LSTMPredictor,
    "exp04_lstm": LSTMPredictor,
    "exp04_lstm_bone": LSTMPredictor,
}

# Add improved predictor if available
if ImprovedLSTMPredictor is not None:
    PREDICTOR_REGISTRY.update({
        "exp05": ImprovedLSTMPredictor,
        "exp05_lstm_improved": ImprovedLSTMPredictor,
    })


def load_predictor(
    experiment: str,
    config_path: Optional[Union[str, Path]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> BasePredictor:
    """Load a predictor for the specified experiment.
    
    Args:
        experiment: Experiment name (e.g., "exp00", "exp04_lstm")
        config_path: Optional custom config path. If None, uses default.
        checkpoint_path: Optional custom checkpoint path. If None, uses config default.
        
    Returns:
        Initialized predictor instance
        
    Examples:
        >>> # Load with default config and checkpoint
        >>> predictor = load_predictor("exp04")
        
        >>> # Load with custom paths
        >>> predictor = load_predictor("exp04", 
        ...     config_path="configs/custom.yaml",
        ...     checkpoint_path="checkpoints/best_model.ckpt")
    """
    if experiment not in PREDICTOR_REGISTRY:
        available = ", ".join(sorted(PREDICTOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown experiment: {experiment}. "
            f"Available experiments: {available}"
        )
    
    # Use default config path if not provided
    if config_path is None:
        # Map experiment to config file
        config_mapping = {
            "exp00": "configs/exp00_cnn.yaml",
            "exp00_cnn": "configs/exp00_cnn.yaml",
            "exp01": "configs/exp01_cnn_lstm.yaml",
            "exp01_cnn_lstm": "configs/exp01_cnn_lstm.yaml",
            "exp02": "configs/exp02_cnn_lstm.yaml",
            "exp02_cnn_lstm": "configs/exp02_cnn_lstm.yaml",
            "exp03": "configs/exp03_lightgbm.yaml",
            "exp03_lightgbm": "configs/exp03_lightgbm.yaml",
            "exp04": "configs/exp04_lstm.yaml",
            "exp04_lstm": "configs/exp04_lstm.yaml",
            "exp04_lstm_bone": "configs/exp04_lstm.yaml",
            "exp05": "configs/exp05_lstm_improved.yaml",
            "exp05_lstm_improved": "configs/exp05_lstm_improved.yaml",
        }
        config_path = config_mapping.get(experiment)
        if config_path is None:
            raise ValueError(f"No default config found for experiment: {experiment}")
    
    # Get predictor class and instantiate
    predictor_cls = PREDICTOR_REGISTRY[experiment]
    return predictor_cls(config_path, checkpoint_path)