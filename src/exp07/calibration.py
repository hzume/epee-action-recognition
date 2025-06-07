"""Temperature Scaling calibration for exp07 models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import log_loss


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibrating neural network predictions.
    
    A single temperature parameter is learned for each output head (left/right).
    """
    
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature_left = nn.Parameter(torch.ones(1) * init_temperature)
        self.temperature_right = nn.Parameter(torch.ones(1) * init_temperature)
    
    def forward(self, logits_left: torch.Tensor, logits_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply temperature scaling to logits.
        
        Args:
            logits_left: Left player logits (batch_size, num_classes)
            logits_right: Right player logits (batch_size, num_classes)
            
        Returns:
            Tuple of calibrated logits for left and right players
        """
        return logits_left / self.temperature_left, logits_right / self.temperature_right
    
    def get_temperatures(self) -> Dict[str, float]:
        """Get current temperature values."""
        return {
            "temperature_left": self.temperature_left.item(),
            "temperature_right": self.temperature_right.item()
        }


class VectorScaling(nn.Module):
    """Vector scaling for calibrating neural network predictions.
    
    Learns a scaling vector (w) and bias vector (b) for each class,
    allowing different calibration for each class independently.
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        # Separate parameters for left and right players
        self.w_left = nn.Parameter(torch.ones(num_classes))
        self.b_left = nn.Parameter(torch.zeros(num_classes))
        self.w_right = nn.Parameter(torch.ones(num_classes))
        self.b_right = nn.Parameter(torch.zeros(num_classes))
        self.num_classes = num_classes
    
    def forward(self, logits_left: torch.Tensor, logits_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply vector scaling to logits.
        
        Args:
            logits_left: Left player logits (batch_size, num_classes)
            logits_right: Right player logits (batch_size, num_classes)
            
        Returns:
            Tuple of calibrated logits for left and right players
        """
        # Element-wise multiplication and addition
        calibrated_left = logits_left * self.w_left + self.b_left
        calibrated_right = logits_right * self.w_right + self.b_right
        
        return calibrated_left, calibrated_right
    
    def get_parameters(self) -> Dict[str, List[float]]:
        """Get current vector scaling parameters."""
        return {
            "w_left": self.w_left.detach().cpu().tolist(),
            "b_left": self.b_left.detach().cpu().tolist(),
            "w_right": self.w_right.detach().cpu().tolist(),
            "b_right": self.b_right.detach().cpu().tolist()
        }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels (n_samples,)
        y_prob: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy and confidence in this bin
            accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
            avg_confidence_in_bin = y_conf[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def learn_temperature_scaling(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01
) -> TemperatureScaling:
    """Learn temperature scaling parameters on validation set.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader (should NOT use balanced sampling)
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate for temperature optimization
        
    Returns:
        Trained TemperatureScaling module
    """
    model.eval()
    temperature_scaling = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temperature_scaling.temperature_left, temperature_scaling.temperature_right], lr=lr, max_iter=max_iter)
    
    # Collect all logits and labels
    all_logits_left = []
    all_logits_right = []
    all_labels_left = []
    all_labels_right = []
    
    print("Collecting validation data...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y_left, y_right, _ = batch
            x = x.to(device)
            
            # Get model predictions
            logits_left, logits_right = model(x)
            
            all_logits_left.append(logits_left.cpu())
            all_logits_right.append(logits_right.cpu())
            all_labels_left.append(y_left.argmax(dim=1).cpu())
            all_labels_right.append(y_right.argmax(dim=1).cpu())
    
    # Concatenate all batches
    all_logits_left = torch.cat(all_logits_left)
    all_logits_right = torch.cat(all_logits_right)
    all_labels_left = torch.cat(all_labels_left)
    all_labels_right = torch.cat(all_labels_right)
    
    # Move to device for optimization
    all_logits_left = all_logits_left.to(device)
    all_logits_right = all_logits_right.to(device)
    all_labels_left = all_labels_left.to(device)
    all_labels_right = all_labels_right.to(device)
    
    print("Optimizing temperature parameters...")
    
    def eval_loss():
        # Apply temperature scaling
        scaled_logits_left, scaled_logits_right = temperature_scaling(all_logits_left, all_logits_right)
        
        # Calculate NLL loss
        loss_left = F.cross_entropy(scaled_logits_left, all_labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, all_labels_right)
        total_loss = (loss_left + loss_right) / 2
        
        return total_loss
    
    # Initial metrics
    with torch.no_grad():
        initial_loss = eval_loss().item()
        
        # Calculate initial ECE
        probs_left = F.softmax(all_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(all_logits_right, dim=1).cpu().numpy()
        labels_left_np = all_labels_left.cpu().numpy()
        labels_right_np = all_labels_right.cpu().numpy()
        
        ece_left_before = expected_calibration_error(labels_left_np, probs_left)
        ece_right_before = expected_calibration_error(labels_right_np, probs_right)
        ece_before = (ece_left_before + ece_right_before) / 2
    
    print(f"Before calibration - Loss: {initial_loss:.4f}, ECE: {ece_before:.4f}")
    
    # Optimize temperature
    def closure():
        optimizer.zero_grad()
        loss = eval_loss()
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    # Final metrics
    with torch.no_grad():
        final_loss = eval_loss().item()
        
        # Apply temperature and calculate final ECE
        scaled_logits_left, scaled_logits_right = temperature_scaling(all_logits_left, all_logits_right)
        probs_left = F.softmax(scaled_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(scaled_logits_right, dim=1).cpu().numpy()
        
        ece_left_after = expected_calibration_error(labels_left_np, probs_left)
        ece_right_after = expected_calibration_error(labels_right_np, probs_right)
        ece_after = (ece_left_after + ece_right_after) / 2
    
    temps = temperature_scaling.get_temperatures()
    print(f"After calibration - Loss: {final_loss:.4f}, ECE: {ece_after:.4f}")
    print(f"Learned temperatures - Left: {temps['temperature_left']:.4f}, Right: {temps['temperature_right']:.4f}")
    
    # Print detailed ECE improvements
    print(f"\nECE improvements:")
    print(f"  Left player:  {ece_left_before:.4f} -> {ece_left_after:.4f} (Δ={ece_left_before - ece_left_after:.4f})")
    print(f"  Right player: {ece_right_before:.4f} -> {ece_right_after:.4f} (Δ={ece_right_before - ece_right_after:.4f})")
    
    return temperature_scaling


def learn_vector_scaling(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    max_iter: int = 50,
    lr: float = 0.01
) -> VectorScaling:
    """Learn vector scaling parameters on validation set.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader (should NOT use balanced sampling)
        device: Device to run on
        num_classes: Number of classes
        max_iter: Maximum optimization iterations
        lr: Learning rate for optimization
        
    Returns:
        Trained VectorScaling module
    """
    model.eval()
    vector_scaling = VectorScaling(num_classes).to(device)
    
    # Use Adam optimizer for vector parameters
    optimizer = torch.optim.Adam([
        vector_scaling.w_left, vector_scaling.b_left,
        vector_scaling.w_right, vector_scaling.b_right
    ], lr=lr)
    
    # Collect all logits and labels
    all_logits_left = []
    all_logits_right = []
    all_labels_left = []
    all_labels_right = []
    
    print("Collecting validation data...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y_left, y_right, _ = batch
            x = x.to(device)
            
            # Get model predictions
            logits_left, logits_right = model(x)
            
            all_logits_left.append(logits_left.cpu())
            all_logits_right.append(logits_right.cpu())
            all_labels_left.append(y_left.argmax(dim=1).cpu())
            all_labels_right.append(y_right.argmax(dim=1).cpu())
    
    # Concatenate all batches
    all_logits_left = torch.cat(all_logits_left).to(device)
    all_logits_right = torch.cat(all_logits_right).to(device)
    all_labels_left = torch.cat(all_labels_left).to(device)
    all_labels_right = torch.cat(all_labels_right).to(device)
    
    print("Optimizing vector scaling parameters...")
    
    # Initial metrics
    with torch.no_grad():
        probs_left = F.softmax(all_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(all_logits_right, dim=1).cpu().numpy()
        labels_left_np = all_labels_left.cpu().numpy()
        labels_right_np = all_labels_right.cpu().numpy()
        
        ece_left_before = expected_calibration_error(labels_left_np, probs_left)
        ece_right_before = expected_calibration_error(labels_right_np, probs_right)
        ece_before = (ece_left_before + ece_right_before) / 2
        
        initial_loss_left = F.cross_entropy(all_logits_left, all_labels_left).item()
        initial_loss_right = F.cross_entropy(all_logits_right, all_labels_right).item()
        initial_loss = (initial_loss_left + initial_loss_right) / 2
    
    print(f"Before calibration - Loss: {initial_loss:.4f}, ECE: {ece_before:.4f}")
    
    # Optimize parameters
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Apply vector scaling
        scaled_logits_left, scaled_logits_right = vector_scaling(all_logits_left, all_logits_right)
        
        # Calculate NLL loss
        loss_left = F.cross_entropy(scaled_logits_left, all_labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, all_labels_right)
        total_loss = (loss_left + loss_right) / 2
        
        # Add L2 regularization to prevent overfitting
        reg_loss = 0.01 * (
            torch.sum((vector_scaling.w_left - 1) ** 2) +
            torch.sum(vector_scaling.b_left ** 2) +
            torch.sum((vector_scaling.w_right - 1) ** 2) +
            torch.sum(vector_scaling.b_right ** 2)
        )
        
        total_loss = total_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{max_iter}, Loss: {total_loss.item():.4f}")
    
    # Final metrics
    with torch.no_grad():
        scaled_logits_left, scaled_logits_right = vector_scaling(all_logits_left, all_logits_right)
        
        probs_left = F.softmax(scaled_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(scaled_logits_right, dim=1).cpu().numpy()
        
        ece_left_after = expected_calibration_error(labels_left_np, probs_left)
        ece_right_after = expected_calibration_error(labels_right_np, probs_right)
        ece_after = (ece_left_after + ece_right_after) / 2
        
        final_loss_left = F.cross_entropy(scaled_logits_left, all_labels_left).item()
        final_loss_right = F.cross_entropy(scaled_logits_right, all_labels_right).item()
        final_loss = (final_loss_left + final_loss_right) / 2
    
    print(f"After calibration - Loss: {final_loss:.4f}, ECE: {ece_after:.4f}")
    print(f"\nECE improvements:")
    print(f"  Left player:  {ece_left_before:.4f} -> {ece_left_after:.4f} (Δ={ece_left_before - ece_left_after:.4f})")
    print(f"  Right player: {ece_right_before:.4f} -> {ece_right_after:.4f} (Δ={ece_right_before - ece_right_after:.4f})")
    
    # Print learned parameters
    params = vector_scaling.get_parameters()
    print(f"\nLearned parameters:")
    print(f"  Left w: {[f'{w:.3f}' for w in params['w_left']]}")
    print(f"  Left b: {[f'{b:.3f}' for b in params['b_left']]}")
    print(f"  Right w: {[f'{w:.3f}' for w in params['w_right']]}")
    print(f"  Right b: {[f'{b:.3f}' for b in params['b_right']]}")
    
    return vector_scaling


def save_temperature_scaling(temperature_scaling: TemperatureScaling, path: Path):
    """Save temperature scaling parameters."""
    temps = temperature_scaling.get_temperatures()
    with open(path, 'w') as f:
        json.dump(temps, f, indent=2)
    print(f"Temperature scaling parameters saved to {path}")


def load_temperature_scaling(path: Path) -> TemperatureScaling:
    """Load temperature scaling parameters."""
    with open(path, 'r') as f:
        temps = json.load(f)
    
    ts = TemperatureScaling()
    ts.temperature_left.data = torch.tensor([temps['temperature_left']])
    ts.temperature_right.data = torch.tensor([temps['temperature_right']])
    
    return ts


def save_vector_scaling(vector_scaling: VectorScaling, path: Path):
    """Save vector scaling parameters."""
    params = vector_scaling.get_parameters()
    params['num_classes'] = vector_scaling.num_classes
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Vector scaling parameters saved to {path}")


def load_vector_scaling(path: Path) -> VectorScaling:
    """Load vector scaling parameters."""
    with open(path, 'r') as f:
        params = json.load(f)
    
    vs = VectorScaling(params['num_classes'])
    vs.w_left.data = torch.tensor(params['w_left'])
    vs.b_left.data = torch.tensor(params['b_left'])
    vs.w_right.data = torch.tensor(params['w_right'])
    vs.b_right.data = torch.tensor(params['b_right'])
    
    return vs


class CalibratedModel(nn.Module):
    """Wrapper that applies calibration (temperature or vector scaling) to model outputs."""
    
    def __init__(self, model: nn.Module, calibration_module: nn.Module):
        super().__init__()
        self.model = model
        self.calibration_module = calibration_module
    
    def forward(self, x):
        # Get original model predictions
        logits_left, logits_right = self.model(x)
        
        # Ensure calibration module is on the same device as logits
        device = logits_left.device
        if next(self.calibration_module.parameters()).device != device:
            self.calibration_module = self.calibration_module.to(device)
        
        # Apply calibration (works for both TemperatureScaling and VectorScaling)
        calibrated_left, calibrated_right = self.calibration_module(logits_left, logits_right)
        
        return calibrated_left, calibrated_right