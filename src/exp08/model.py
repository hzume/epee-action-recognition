"""Model architectures for exp08 - 3D pose-based action recognition"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np
import math

from .utils import Config3D, create_joint_connectivity_matrix


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for joint relationships"""
    
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self.proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Joint connectivity for skeletal structure
        self.register_buffer('connectivity', create_joint_connectivity_matrix())
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_joints, features)
        Returns:
            Attended features of same shape
        """
        B, T, J, D = x.shape
        
        # Reshape for attention: (B*T, J, D)
        x_flat = x.view(B * T, J, D)
        
        # Generate Q, K, V
        qkv = self.qkv(x_flat).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B * T, J, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply skeletal connectivity mask
        connectivity_mask = self.connectivity.unsqueeze(0).unsqueeze(0)  # (1, 1, J, J)
        attn = attn.masked_fill(connectivity_mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn @ v  # (B*T, num_heads, J, head_dim)
        out = out.transpose(1, 2).contiguous().view(B * T, J, D)
        out = self.proj(out)
        
        # Reshape back
        return out.view(B, T, J, D)


class JointEmbedding(nn.Module):
    """Embedding layer for joint-specific features"""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.joint_embeddings = nn.ModuleList([
            nn.Linear(input_dim, embedding_dim) for _ in range(17)
        ])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_joints, input_dim)
        Returns:
            Joint embeddings (batch_size, seq_len, num_joints, embedding_dim)
        """
        B, T, J, D = x.shape
        
        # Apply joint-specific embeddings
        embedded = []
        for j in range(J):
            joint_features = x[:, :, j, :]  # (B, T, D)
            joint_embedded = self.joint_embeddings[j](joint_features)  # (B, T, embedding_dim)
            embedded.append(joint_embedded)
        
        embedded = torch.stack(embedded, dim=2)  # (B, T, J, embedding_dim)
        embedded = self.layer_norm(embedded)
        
        return embedded


class ThreeDPoseLSTMModel(nn.Module):
    """3D Pose-based LSTM model with spatial and temporal attention"""
    
    def __init__(self, config: Config3D):
        super().__init__()
        self.config = config
        
        # Calculate input dimensions
        self.pose_dim = 17 * 3  # 17 joints × 3 coordinates
        
        # Determine additional feature dimension
        # This will be set during first forward pass
        self.additional_feature_dim = 0
        self.feature_dim_computed = False
        
        # Joint embedding
        if config.model["use_joint_embeddings"]:
            self.joint_embedding = JointEmbedding(
                input_dim=3,  # x, y, z coordinates
                embedding_dim=config.model["joint_embedding_dim"]
            )
            joint_feature_dim = config.model["joint_embedding_dim"]
        else:
            self.joint_embedding = None
            joint_feature_dim = 3
        
        # Spatial attention
        if config.model["use_spatial_attention"]:
            self.spatial_attention = SpatialAttention(
                input_dim=joint_feature_dim,
                num_heads=config.model["spatial_attention_heads"]
            )
        else:
            self.spatial_attention = None
        
        # Feature processing layers (will be initialized after first forward pass)
        self.feature_projector = None
        self.input_norm = None
        
        # LSTM layers
        self.lstm_input_dim = 17 * joint_feature_dim  # Will be updated with additional features
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=config.model["hidden_size"],
            num_layers=config.model["num_layers"],
            batch_first=True,
            dropout=config.model["dropout"] if config.model["num_layers"] > 1 else 0,
            bidirectional=config.model["bidirectional"]
        )
        
        # Output dimension after LSTM
        lstm_output_size = config.model["hidden_size"] * (2 if config.model["bidirectional"] else 1)
        
        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(lstm_output_size, config.model["hidden_size"]),
            nn.Tanh(),
            nn.Linear(config.model["hidden_size"], 1)
        )
        
        # Classification heads
        self.fc_left = nn.Sequential(
            nn.Linear(lstm_output_size, config.model["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(config.model["dropout"]),
            nn.Linear(config.model["hidden_size"], config.num_classes)
        )
        
        self.fc_right = nn.Sequential(
            nn.Linear(lstm_output_size, config.model["hidden_size"]),
            nn.ReLU(),
            nn.Dropout(config.model["dropout"]),
            nn.Linear(config.model["hidden_size"], config.num_classes)
        )
    
    def _initialize_feature_processing(self, total_feature_dim: int):
        """Initialize feature processing layers after determining input dimension"""
        self.lstm.input_size = total_feature_dim
        
        # Reinitialize LSTM with correct input size
        self.lstm = nn.LSTM(
            input_size=total_feature_dim,
            hidden_size=self.config.model["hidden_size"],
            num_layers=self.config.model["num_layers"],
            batch_first=True,
            dropout=self.config.model["dropout"] if self.config.model["num_layers"] > 1 else 0,
            bidirectional=self.config.model["bidirectional"]
        ).to(next(self.parameters()).device)
        
        # Feature projection for additional features
        if self.additional_feature_dim > 0:
            self.feature_projector = nn.Linear(
                self.additional_feature_dim, 
                self.additional_feature_dim
            ).to(next(self.parameters()).device)
        
        # Input normalization
        self.input_norm = nn.LayerNorm(total_feature_dim).to(next(self.parameters()).device)
        
        self.feature_dim_computed = True
    
    def _process_3d_poses(self, pose_sequence):
        """Process 3D pose sequence through spatial components"""
        B, T, _ = pose_sequence.shape
        
        # Reshape to (B, T, 17, 3)
        poses_3d = pose_sequence[:, :, :self.pose_dim].view(B, T, 17, 3)
        
        # Joint embeddings
        if self.joint_embedding is not None:
            poses_3d = self.joint_embedding(poses_3d)
        
        # Spatial attention
        if self.spatial_attention is not None:
            poses_3d = self.spatial_attention(poses_3d)
        
        # Flatten back to (B, T, features)
        poses_flat = poses_3d.view(B, T, -1)
        
        return poses_flat
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, features)
            
        Returns:
            Tuple of (left_logits, right_logits)
        """
        B, T, total_features = x.shape
        
        # Initialize feature processing if not done
        if not self.feature_dim_computed:
            self.additional_feature_dim = total_features - self.pose_dim
            joint_feature_dim = self.config.model["joint_embedding_dim"] if self.joint_embedding else 3
            final_feature_dim = 17 * joint_feature_dim + self.additional_feature_dim
            self._initialize_feature_processing(final_feature_dim)
        
        # Split pose and additional features
        pose_features = x[:, :, :self.pose_dim]
        additional_features = x[:, :, self.pose_dim:] if self.additional_feature_dim > 0 else None
        
        # Process 3D poses
        pose_processed = self._process_3d_poses(pose_features)
        
        # Combine with additional features
        if additional_features is not None and self.feature_projector is not None:
            additional_processed = self.feature_projector(additional_features)
            combined_features = torch.cat([pose_processed, additional_processed], dim=-1)
        else:
            combined_features = pose_processed
        
        # Normalize input
        if self.input_norm is not None:
            combined_features = self.input_norm(combined_features)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(combined_features)
        
        # Temporal attention
        attention_weights = F.softmax(self.temporal_attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        y_left = self.fc_left(context)
        y_right = self.fc_right(context)
        
        return y_left, y_right


class ThreeDLitModel(L.LightningModule):
    """PyTorch Lightning wrapper for 3D pose model"""
    
    def __init__(self, config: Config3D, num_training_steps: int = None):
        super().__init__()
        self.config = config
        self.num_training_steps = num_training_steps
        
        # Initialize model
        self.model = ThreeDPoseLSTMModel(config)
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Class weights for handling imbalanced data
        class_weights = torch.ones(config.num_classes)
        for class_name, weight in config.loss["class_weights"].items():
            if class_name in config.metadata["action_to_id"]:
                class_id = config.metadata["action_to_id"][class_name]
                class_weights[class_id] = weight
            elif class_name == "none":
                class_weights[0] = weight
        
        self.register_buffer('class_weights', class_weights)
        
        # Loss weights
        self.temporal_consistency_weight = config.loss["temporal_consistency_weight"]
        self.spatial_consistency_weight = config.loss["spatial_consistency_weight"]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y_left, y_right, info = batch
        y_left_hat, y_right_hat = self.model(x)
        
        # Primary classification loss
        loss_left = F.cross_entropy(
            y_left_hat,
            y_left.argmax(dim=1),
            weight=self.class_weights
        )
        loss_right = F.cross_entropy(
            y_right_hat,
            y_right.argmax(dim=1),
            weight=self.class_weights
        )
        primary_loss = (loss_left + loss_right) / 2
        
        # Temporal consistency loss (if configured)
        temporal_loss = torch.tensor(0.0, device=self.device)
        if self.temporal_consistency_weight > 0:
            # Add temporal smoothness constraint
            pred_probs_left = F.softmax(y_left_hat, dim=1)
            pred_probs_right = F.softmax(y_right_hat, dim=1)
            
            # This would require sequence-level batching for proper implementation
            # For now, we use a simplified version
            temporal_loss = 0.0
        
        # Total loss
        total_loss = primary_loss + self.temporal_consistency_weight * temporal_loss
        
        # Log metrics
        self.log("train/loss", total_loss, prog_bar=True, on_epoch=True)
        self.log("train/loss_left", loss_left)
        self.log("train/loss_right", loss_right)
        if self.temporal_consistency_weight > 0:
            self.log("train/temporal_loss", temporal_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y_left, y_right, info = batch
        y_left_hat, y_right_hat = self.model(x)
        
        # Loss
        loss_left = F.cross_entropy(y_left_hat, y_left.argmax(dim=1))
        loss_right = F.cross_entropy(y_right_hat, y_right.argmax(dim=1))
        loss = (loss_left + loss_right) / 2
        
        # Predictions
        pred_left = y_left_hat.argmax(dim=1)
        pred_right = y_right_hat.argmax(dim=1)
        true_left = y_left.argmax(dim=1)
        true_right = y_right.argmax(dim=1)
        
        # Accuracy
        acc_left = (pred_left == true_left).float().mean()
        acc_right = (pred_right == true_right).float().mean()
        acc = (acc_left + acc_right) / 2
        
        # Accuracy ignoring 'none' class
        mask_left = true_left != 0
        mask_right = true_right != 0
        
        acc_left_ignore_none = (pred_left[mask_left] == true_left[mask_left]).float().mean() if mask_left.sum() > 0 else torch.tensor(0.0)
        acc_right_ignore_none = (pred_right[mask_right] == true_right[mask_right]).float().mean() if mask_right.sum() > 0 else torch.tensor(0.0)
        acc_ignore_none = (acc_left_ignore_none + acc_right_ignore_none) / 2
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/acc_ignore_none", acc_ignore_none, prog_bar=True)
        
        # Store predictions for epoch-end metrics
        if not hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs = []
        
        self.validation_step_outputs.append({
            'pred_left': pred_left.cpu(),
            'pred_right': pred_right.cpu(),
            'true_left': true_left.cpu(),
            'true_right': true_right.cpu()
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if not hasattr(self, 'validation_step_outputs') or not self.validation_step_outputs:
            return
        
        # Gather all predictions
        all_pred_left = torch.cat([x['pred_left'] for x in self.validation_step_outputs])
        all_pred_right = torch.cat([x['pred_right'] for x in self.validation_step_outputs])
        all_true_left = torch.cat([x['true_left'] for x in self.validation_step_outputs])
        all_true_right = torch.cat([x['true_right'] for x in self.validation_step_outputs])
        
        # Calculate F1 scores
        f1_left = f1_score(
            all_true_left.numpy(),
            all_pred_left.numpy(),
            average='macro',
            zero_division=0
        )
        f1_right = f1_score(
            all_true_right.numpy(),
            all_pred_right.numpy(),
            average='macro',
            zero_division=0
        )
        f1_macro = (f1_left + f1_right) / 2
        
        self.log("val/f1_macro", f1_macro, prog_bar=True)
        self.log("val/f1_left", f1_left)
        self.log("val/f1_right", f1_right)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def predict_step(self, batch, batch_idx):
        x, y_left, y_right, info = batch
        y_left_hat, y_right_hat = self.model(x)
        return y_left_hat, y_right_hat
    
    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training["learning_rate"],
            weight_decay=self.config.training["weight_decay"]
        )
        
        # Learning rate scheduler
        if self.num_training_steps and self.config.training["scheduler"] == "cosine":
            warmup_steps = int(self.config.training["warmup_steps"] * self.num_training_steps)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.num_training_steps
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }
            }
        
        return optimizer