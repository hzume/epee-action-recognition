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
    learn_temperature_scaling_f1,
    learn_vector_scaling,
    learn_vector_scaling_f1,
    save_temperature_scaling,
    save_vector_scaling,
    load_temperature_scaling,
    load_vector_scaling,
    expected_calibration_error,
    sample_predictions_with_temperature,
    apply_calibration_to_ensemble_logits,
    learn_temperature_scaling_on_logits,
    learn_temperature_scaling_f1_on_logits,
    learn_vector_scaling_on_logits,
    learn_vector_scaling_f1_on_logits
)
from .dataset import EpeeDataset
from .model import ImprovedLSTMModel, LitModel
from .utils import Config

__all__ = [
    'TemperatureScaling',
    'VectorScaling',
    'CalibratedModel',
    'learn_temperature_scaling',
    'learn_temperature_scaling_f1',
    'learn_vector_scaling',
    'learn_vector_scaling_f1',
    'save_temperature_scaling',
    'save_vector_scaling',
    'load_temperature_scaling',
    'load_vector_scaling',
    'expected_calibration_error',
    'sample_predictions_with_temperature',
    'apply_calibration_to_ensemble_logits',
    'learn_temperature_scaling_on_logits',
    'learn_temperature_scaling_f1_on_logits',
    'learn_vector_scaling_on_logits',
    'learn_vector_scaling_f1_on_logits',
    'EpeeDataset',
    'ImprovedLSTMModel',
    'LitModel',
    'Config'
]