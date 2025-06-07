"""
Exp08: 3D Pose-based LSTM model for fencing action recognition
- Uses 3D pose data instead of 2D projections
- Incorporates depth information and 3D spatial relationships
- Enhanced with 3D-specific features like joint angles and orientations
- Includes temporal modeling for action sequences
"""

from .model import ThreeDPoseLSTMModel, ThreeDLitModel
from .dataset import ThreeDPoseDataset
from .utils import Config3D

__all__ = [
    'ThreeDPoseLSTMModel',
    'ThreeDLitModel', 
    'ThreeDPoseDataset',
    'Config3D'
]