"""
Exp07: LSTM with frame difference features and temperature scaling calibration
- Based on exp06 with additional frame-to-frame motion features
- Includes temperature scaling for probability calibration
- Handles class imbalance between training and inference
"""

from .calibration import (
    TemperatureScaling,
    CalibratedModel,
    learn_temperature_scaling,
    save_temperature_scaling,
    load_temperature_scaling,
    expected_calibration_error
)

__all__ = [
    'TemperatureScaling',
    'CalibratedModel', 
    'learn_temperature_scaling',
    'save_temperature_scaling',
    'load_temperature_scaling',
    'expected_calibration_error'
]