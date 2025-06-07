"""
Exp07: LSTM with frame difference features and calibration
- Based on exp06 with additional frame-to-frame motion features
- Includes temperature scaling and vector scaling for probability calibration
- Handles class imbalance between training and inference
"""

from .calibration import (
    TemperatureScaling,
    VectorScaling,
    CalibratedModel,
    learn_temperature_scaling,
    learn_vector_scaling,
    save_temperature_scaling,
    save_vector_scaling,
    load_temperature_scaling,
    load_vector_scaling,
    expected_calibration_error
)
from .dataset import EpeeDataset
from .model import ImprovedLSTMModel, LitModel
from .utils import Config

__all__ = [
    'TemperatureScaling',
    'VectorScaling',
    'CalibratedModel', 
    'learn_temperature_scaling',
    'learn_vector_scaling',
    'save_temperature_scaling',
    'save_vector_scaling',
    'load_temperature_scaling',
    'load_vector_scaling',
    'expected_calibration_error',
    'EpeeDataset',
    'ImprovedLSTMModel',
    'LitModel',
    'Config'
]