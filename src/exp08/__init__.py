"""
Exp08: 3D Pose-based LSTM model for fencing action recognition
- Uses 3D pose data instead of 2D projections
- Incorporates depth information and 3D spatial relationships
- Enhanced with 3D-specific features like joint angles and orientations
- Includes spatial attention for joint relationships
- Temporal modeling with LSTM for action sequences
"""

from .model import ThreeDPoseLSTMModel, ThreeDLitModel, SpatialAttention, JointEmbedding
from .dataset import ThreeDPoseDataset, create_3d_pose_datasets
from .utils import (
    Config3D, 
    compute_3d_angle, 
    apply_3d_rotation, 
    normalize_3d_pose,
    evaluate_3d_predictions,
    print_3d_classification_report,
    save_3d_predictions,
    create_joint_connectivity_matrix
)

__all__ = [
    'ThreeDPoseLSTMModel',
    'ThreeDLitModel',
    'SpatialAttention',
    'JointEmbedding',
    'ThreeDPoseDataset',
    'create_3d_pose_datasets',
    'Config3D',
    'compute_3d_angle',
    'apply_3d_rotation', 
    'normalize_3d_pose',
    'evaluate_3d_predictions',
    'print_3d_classification_report',
    'save_3d_predictions',
    'create_joint_connectivity_matrix'
]