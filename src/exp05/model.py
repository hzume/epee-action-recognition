"""Model module for exp05"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
import numpy as np

from .utils import Config


class ImprovedLSTMModel(nn.Module):
    """Improved LSTM model for fencing action recognition"""
    
    def __init__(
        self,
        num_features: int,
        config: Config
    ):
        super().__init__()
        self.config = config
        
        # Input normalization (additional layer normalization after dataset normalization)
        self.input_norm = nn.LayerNorm(num_features)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Output dimension after LSTM
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        
        # Classification heads
        self.fc_left = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
        self.fc_right = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        
        Returns:
            Tuple of (left_logits, right_logits)
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Extract features
        x = self.feature_extractor(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        y_left = self.fc_left(context)
        y_right = self.fc_right(context)
        
        return y_left, y_right


class LitModel(L.LightningModule):
    """PyTorch Lightning wrapper for the model"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        num_training_steps: int = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.num_training_steps = num_training_steps
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Class weights for handling imbalanced data
        self.register_buffer(
            'class_weights',
            torch.ones(config.num_classes)
        )
        # Reduce weight for 'none' class
        self.class_weights[0] = 0.1
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y_left, y_right = batch
        y_left_hat, y_right_hat = self.model(x)
        
        # Weighted cross entropy loss
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
        loss = (loss_left + loss_right) / 2
        
        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_left", loss_left)
        self.log("train/loss_right", loss_right)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_left, y_right = batch
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
        
        if mask_left.sum() > 0:
            acc_left_ignore_none = (pred_left[mask_left] == true_left[mask_left]).float().mean()
        else:
            acc_left_ignore_none = torch.tensor(0.0)
            
        if mask_right.sum() > 0:
            acc_right_ignore_none = (pred_right[mask_right] == true_right[mask_right]).float().mean()
        else:
            acc_right_ignore_none = torch.tensor(0.0)
            
        acc_ignore_none = (acc_left_ignore_none + acc_right_ignore_none) / 2
        
        # Log metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/acc_ignore_none", acc_ignore_none, prog_bar=True)
        
        # Store predictions for epoch-end metrics
        self.validation_step_outputs.append({
            'pred_left': pred_left.cpu(),
            'pred_right': pred_right.cpu(),
            'true_left': true_left.cpu(),
            'true_right': true_right.cpu()
        })
        
        return loss
    
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
    
    def on_validation_epoch_end(self):
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
        x, y_left, y_right = batch
        y_left_hat, y_right_hat = self.model(x)
        return y_left_hat, y_right_hat
    
    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        if self.num_training_steps:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * self.num_training_steps),
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