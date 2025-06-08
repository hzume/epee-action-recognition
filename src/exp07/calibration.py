"""Temperature Scaling calibration for exp07 models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import log_loss, f1_score
from scipy.optimize import minimize
import random


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


class DistributionCalibration:
    """Distribution calibration using optimized thresholds for each class.
    
    This class implements a two-stage calibration approach:
    1. First apply probability calibration (Temperature/Vector Scaling)
    2. Then adjust class thresholds to match target label distribution
    """
    
    def __init__(self, num_classes: int, target_distribution: Optional[Dict[int, float]] = None):
        """Initialize distribution calibration.
        
        Args:
            num_classes: Number of classes
            target_distribution: Target distribution for each class {class_id: frequency}
                                If None, will use uniform distribution
        """
        self.num_classes = num_classes
        self.target_distribution = target_distribution or {i: 1.0/num_classes for i in range(num_classes)}
        
        # Learned thresholds for each class (left and right players)
        self.thresholds_left = np.zeros(num_classes)
        self.thresholds_right = np.zeros(num_classes)
        
        # Default threshold is 1/num_classes (uniform prediction)
        default_threshold = 1.0 / num_classes
        self.thresholds_left.fill(default_threshold)
        self.thresholds_right.fill(default_threshold)
    
    def predict_with_thresholds(self, probs: np.ndarray, side: str) -> np.ndarray:
        """Apply threshold-based prediction.
        
        Args:
            probs: Predicted probabilities (N, num_classes)
            side: 'left' or 'right' player
            
        Returns:
            Predicted class indices
        """
        thresholds = self.thresholds_left if side == 'left' else self.thresholds_right
        
        # For each sample, check if any class probability exceeds its threshold
        predictions = np.zeros(len(probs), dtype=int)
        
        for i, prob in enumerate(probs):
            # Find classes that exceed their thresholds
            exceeding_classes = np.where(prob >= thresholds)[0]
            
            if len(exceeding_classes) > 0:
                # Choose class with highest probability among those exceeding threshold
                best_class = exceeding_classes[np.argmax(prob[exceeding_classes])]
                predictions[i] = best_class
            else:
                # If no class exceeds threshold, choose class with highest probability
                predictions[i] = np.argmax(prob)
        
        return predictions
    
    def get_parameters(self) -> Dict[str, List[float]]:
        """Get current threshold parameters."""
        return {
            "thresholds_left": self.thresholds_left.tolist(),
            "thresholds_right": self.thresholds_right.tolist(),
            "target_distribution": self.target_distribution,
            "num_classes": self.num_classes
        }


def sample_predictions_with_temperature(
    logits: torch.Tensor, 
    temperature: float = 1.0, 
    random_seed: Optional[int] = None
) -> torch.Tensor:
    """Sample predictions from logits using temperature scaling.
    
    Args:
        logits: Model logits (batch_size, num_classes)
        temperature: Temperature parameter for sampling (1.0 = standard sampling, >1.0 = more uniform, <1.0 = more peaked)
        random_seed: Random seed for reproducible sampling
        
    Returns:
        Sampled class indices (batch_size,)
    """
    if random_seed is not None:
        # Set random seed for reproducible sampling
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    # Apply temperature scaling to logits
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=1)
    
    # Sample from the probability distribution
    # torch.multinomial samples from categorical distribution
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(1)
    
    return sampled_indices


def apply_calibration_to_ensemble_logits(
    ensemble_logits_left: torch.Tensor,
    ensemble_logits_right: torch.Tensor,
    calibration_module: Optional[nn.Module] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply calibration to ensemble-averaged logits.
    
    Args:
        ensemble_logits_left: Ensemble-averaged left player logits (batch_size, num_classes)
        ensemble_logits_right: Ensemble-averaged right player logits (batch_size, num_classes)
        calibration_module: Calibration module (TemperatureScaling or VectorScaling)
        
    Returns:
        Tuple of calibrated logits for left and right players
    """
    if calibration_module is None:
        # No calibration, return original logits
        return ensemble_logits_left, ensemble_logits_right
    
    # Apply calibration
    calibrated_left, calibrated_right = calibration_module(ensemble_logits_left, ensemble_logits_right)
    
    return calibrated_left, calibrated_right


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
    max_iter: int = 200,
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


def learn_temperature_scaling_f1(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 0.01
) -> TemperatureScaling:
    """Learn temperature scaling parameters optimizing for F1 score.
    
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
    
    # Use Adam optimizer for F1-based optimization (more stable than LBFGS for non-smooth objectives)
    optimizer = torch.optim.Adam([temperature_scaling.temperature_left, temperature_scaling.temperature_right], lr=lr)
    
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
    
    print("Optimizing temperature parameters for F1 score...")
    
    def calculate_f1_score():
        # Apply temperature scaling
        scaled_logits_left, scaled_logits_right = temperature_scaling(all_logits_left, all_logits_right)
        
        # Get predictions
        pred_left = scaled_logits_left.argmax(dim=1).cpu().numpy()
        pred_right = scaled_logits_right.argmax(dim=1).cpu().numpy()
        
        # Convert labels to numpy
        labels_left_np = all_labels_left.cpu().numpy()
        labels_right_np = all_labels_right.cpu().numpy()
        
        # Calculate F1 scores
        f1_left = f1_score(labels_left_np, pred_left, average='macro', zero_division=0)
        f1_right = f1_score(labels_right_np, pred_right, average='macro', zero_division=0)
        f1_avg = (f1_left + f1_right) / 2
        
        return f1_avg, f1_left, f1_right
    
    # Initial metrics
    with torch.no_grad():
        # Calculate initial ECE for comparison
        probs_left = F.softmax(all_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(all_logits_right, dim=1).cpu().numpy()
        labels_left_np = all_labels_left.cpu().numpy()
        labels_right_np = all_labels_right.cpu().numpy()
        
        ece_left_before = expected_calibration_error(labels_left_np, probs_left)
        ece_right_before = expected_calibration_error(labels_right_np, probs_right)
        ece_before = (ece_left_before + ece_right_before) / 2
        
        # Initial F1 score
        f1_before, f1_left_before, f1_right_before = calculate_f1_score()
    
    print(f"Before calibration - F1: {f1_before:.4f} (L:{f1_left_before:.4f}, R:{f1_right_before:.4f}), ECE: {ece_before:.4f}")
    
    # Optimize temperature to maximize F1 score
    best_f1 = f1_before
    best_temperatures = (temperature_scaling.temperature_left.item(), temperature_scaling.temperature_right.item())
    
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Calculate F1 score (note: F1 is not differentiable, so we use a proxy)
        # Use negative cross-entropy as a differentiable proxy for F1 optimization
        scaled_logits_left, scaled_logits_right = temperature_scaling(all_logits_left, all_logits_right)
        
        # Calculate weighted cross-entropy loss (inverse class frequency weighting)
        # This helps with F1 optimization for imbalanced data
        loss_left = F.cross_entropy(scaled_logits_left, all_labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, all_labels_right)
        loss = (loss_left + loss_right) / 2
        
        # Add regularization to prevent extreme temperature values
        reg_loss = 0.001 * (
            torch.abs(temperature_scaling.temperature_left - 1.0) + 
            torch.abs(temperature_scaling.temperature_right - 1.0)
        )
        
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        
        # Check F1 score every few iterations
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                f1_current, f1_left_current, f1_right_current = calculate_f1_score()
                if f1_current > best_f1:
                    best_f1 = f1_current
                    best_temperatures = (temperature_scaling.temperature_left.item(), temperature_scaling.temperature_right.item())
                
                print(f"Iteration {i+1}/{max_iter}, F1: {f1_current:.4f} (L:{f1_left_current:.4f}, R:{f1_right_current:.4f}), Loss: {total_loss.item():.4f}")
    
    # Set best temperatures
    temperature_scaling.temperature_left.data = torch.tensor([best_temperatures[0]])
    temperature_scaling.temperature_right.data = torch.tensor([best_temperatures[1]])
    
    # Final metrics
    with torch.no_grad():
        f1_after, f1_left_after, f1_right_after = calculate_f1_score()
        
        # Apply temperature and calculate final ECE
        scaled_logits_left, scaled_logits_right = temperature_scaling(all_logits_left, all_logits_right)
        probs_left = F.softmax(scaled_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(scaled_logits_right, dim=1).cpu().numpy()
        
        ece_left_after = expected_calibration_error(labels_left_np, probs_left)
        ece_right_after = expected_calibration_error(labels_right_np, probs_right)
        ece_after = (ece_left_after + ece_right_after) / 2
    
    temps = temperature_scaling.get_temperatures()
    print(f"After calibration - F1: {f1_after:.4f} (L:{f1_left_after:.4f}, R:{f1_right_after:.4f}), ECE: {ece_after:.4f}")
    print(f"Learned temperatures - Left: {temps['temperature_left']:.4f}, Right: {temps['temperature_right']:.4f}")
    
    # Print improvements
    print(f"\nF1 improvements:")
    print(f"  Left player:  {f1_left_before:.4f} -> {f1_left_after:.4f} (Δ={f1_left_after - f1_left_before:.4f})")
    print(f"  Right player: {f1_right_before:.4f} -> {f1_right_after:.4f} (Δ={f1_right_after - f1_right_before:.4f})")
    print(f"  Average:      {f1_before:.4f} -> {f1_after:.4f} (Δ={f1_after - f1_before:.4f})")
    
    print(f"\nECE impact:")
    print(f"  ECE: {ece_before:.4f} -> {ece_after:.4f} (Δ={ece_before - ece_after:.4f})")
    
    return temperature_scaling


def learn_vector_scaling(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    max_iter: int = 200,
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


def learn_vector_scaling_f1(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    max_iter: int = 200,
    lr: float = 0.01
) -> VectorScaling:
    """Learn vector scaling parameters optimizing for F1 score.
    
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
    
    print("Optimizing vector scaling parameters for F1 score...")
    
    def calculate_f1_score():
        # Apply vector scaling
        scaled_logits_left, scaled_logits_right = vector_scaling(all_logits_left, all_logits_right)
        
        # Get predictions
        pred_left = scaled_logits_left.argmax(dim=1).cpu().numpy()
        pred_right = scaled_logits_right.argmax(dim=1).cpu().numpy()
        
        # Convert labels to numpy
        labels_left_np = all_labels_left.cpu().numpy()
        labels_right_np = all_labels_right.cpu().numpy()
        
        # Calculate F1 scores
        f1_left = f1_score(labels_left_np, pred_left, average='macro', zero_division=0)
        f1_right = f1_score(labels_right_np, pred_right, average='macro', zero_division=0)
        f1_avg = (f1_left + f1_right) / 2
        
        return f1_avg, f1_left, f1_right
    
    # Initial metrics
    with torch.no_grad():
        probs_left = F.softmax(all_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(all_logits_right, dim=1).cpu().numpy()
        labels_left_np = all_labels_left.cpu().numpy()
        labels_right_np = all_labels_right.cpu().numpy()
        
        ece_left_before = expected_calibration_error(labels_left_np, probs_left)
        ece_right_before = expected_calibration_error(labels_right_np, probs_right)
        ece_before = (ece_left_before + ece_right_before) / 2
        
        # Initial F1 score
        f1_before, f1_left_before, f1_right_before = calculate_f1_score()
    
    print(f"Before calibration - F1: {f1_before:.4f} (L:{f1_left_before:.4f}, R:{f1_right_before:.4f}), ECE: {ece_before:.4f}")
    
    # Track best F1 score and parameters
    best_f1 = f1_before
    best_state_dict = vector_scaling.state_dict().copy()
    
    # Optimize parameters
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Apply vector scaling
        scaled_logits_left, scaled_logits_right = vector_scaling(all_logits_left, all_logits_right)
        
        # Calculate cross-entropy loss as proxy for F1 optimization
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
        
        # Check F1 score every few iterations and save best
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                f1_current, f1_left_current, f1_right_current = calculate_f1_score()
                if f1_current > best_f1:
                    best_f1 = f1_current
                    best_state_dict = vector_scaling.state_dict().copy()
                
                print(f"Iteration {i+1}/{max_iter}, F1: {f1_current:.4f} (L:{f1_left_current:.4f}, R:{f1_right_current:.4f}), Loss: {total_loss.item():.4f}")
    
    # Load best parameters
    vector_scaling.load_state_dict(best_state_dict)
    
    # Final metrics
    with torch.no_grad():
        f1_after, f1_left_after, f1_right_after = calculate_f1_score()
        
        scaled_logits_left, scaled_logits_right = vector_scaling(all_logits_left, all_logits_right)
        probs_left = F.softmax(scaled_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(scaled_logits_right, dim=1).cpu().numpy()
        
        ece_left_after = expected_calibration_error(labels_left_np, probs_left)
        ece_right_after = expected_calibration_error(labels_right_np, probs_right)
        ece_after = (ece_left_after + ece_right_after) / 2
    
    print(f"After calibration - F1: {f1_after:.4f} (L:{f1_left_after:.4f}, R:{f1_right_after:.4f}), ECE: {ece_after:.4f}")
    
    # Print improvements
    print(f"\nF1 improvements:")
    print(f"  Left player:  {f1_left_before:.4f} -> {f1_left_after:.4f} (Δ={f1_left_after - f1_left_before:.4f})")
    print(f"  Right player: {f1_right_before:.4f} -> {f1_right_after:.4f} (Δ={f1_right_after - f1_right_before:.4f})")
    print(f"  Average:      {f1_before:.4f} -> {f1_after:.4f} (Δ={f1_after - f1_before:.4f})")
    
    print(f"\nECE impact:")
    print(f"  ECE: {ece_before:.4f} -> {ece_after:.4f} (Δ={ece_before - ece_after:.4f})")
    
    # Print learned parameters
    params = vector_scaling.get_parameters()
    print(f"\nLearned parameters:")
    print(f"  Left w: {[f'{w:.3f}' for w in params['w_left']]}")
    print(f"  Left b: {[f'{b:.3f}' for b in params['b_left']]}")
    print(f"  Right w: {[f'{w:.3f}' for w in params['w_right']]}")
    print(f"  Right b: {[f'{b:.3f}' for b in params['b_right']]}")
    
    return vector_scaling


def learn_distribution_calibration(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
    target_distribution: Optional[Dict[int, float]] = None,
    prob_calibration_module: Optional[torch.nn.Module] = None
) -> DistributionCalibration:
    """Learn distribution calibration thresholds.
    
    Args:
        model: Trained model
        dataloader: Validation dataloader (should NOT use balanced sampling)
        device: Device to run on
        num_classes: Number of classes
        target_distribution: Target distribution {class_id: frequency}
        prob_calibration_module: Optional probability calibration (Temperature/Vector Scaling)
        
    Returns:
        Trained DistributionCalibration object
    """
    model.eval()
    
    # Collect all probabilities and labels
    all_probs_left = []
    all_probs_right = []
    all_labels_left = []
    all_labels_right = []
    
    print("Collecting validation data for distribution calibration...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y_left, y_right, _ = batch
            x = x.to(device)
            
            # Get model predictions
            logits_left, logits_right = model(x)
            
            # Apply probability calibration if provided
            if prob_calibration_module is not None:
                prob_calibration_module = prob_calibration_module.to(device)
                logits_left, logits_right = prob_calibration_module(logits_left, logits_right)
            
            # Convert to probabilities
            probs_left = F.softmax(logits_left, dim=1).cpu().numpy()
            probs_right = F.softmax(logits_right, dim=1).cpu().numpy()
            
            all_probs_left.append(probs_left)
            all_probs_right.append(probs_right)
            all_labels_left.append(y_left.argmax(dim=1).cpu().numpy())
            all_labels_right.append(y_right.argmax(dim=1).cpu().numpy())
    
    # Concatenate all batches
    all_probs_left = np.vstack(all_probs_left)
    all_probs_right = np.vstack(all_probs_right)
    all_labels_left = np.concatenate(all_labels_left)
    all_labels_right = np.concatenate(all_labels_right)
    
    # Analyze current distribution
    print("\n=== Current Prediction Distribution Analysis ===")
    current_pred_left = np.argmax(all_probs_left, axis=1)
    current_pred_right = np.argmax(all_probs_right, axis=1)
    
    print("Left player current distribution:")
    for class_id in range(num_classes):
        current_freq = (current_pred_left == class_id).mean()
        print(f"  Class {class_id}: {current_freq:.3f}")
    
    print("Right player current distribution:")
    for class_id in range(num_classes):
        current_freq = (current_pred_right == class_id).mean()
        print(f"  Class {class_id}: {current_freq:.3f}")
    
    # Set target distribution if not provided
    if target_distribution is None:
        # Use actual label distribution from validation data
        target_distribution = {}
        all_labels = np.concatenate([all_labels_left, all_labels_right])
        for class_id in range(num_classes):
            target_distribution[class_id] = (all_labels == class_id).mean()
        
        print("\nUsing actual validation label distribution as target:")
        for class_id, freq in target_distribution.items():
            print(f"  Class {class_id}: {freq:.3f}")
    
    # Initialize distribution calibration
    dist_cal = DistributionCalibration(num_classes, target_distribution)
    
    def optimize_thresholds_for_side(probs: np.ndarray, side: str) -> np.ndarray:
        """Optimize thresholds for one side (left or right)."""
        
        # Calculate current prediction distribution
        current_preds = np.argmax(probs, axis=1)
        current_dist = np.array([(current_preds == i).mean() for i in range(num_classes)])
        target_dist = np.array([target_distribution[i] for i in range(num_classes)])
        
        print(f"  Current distribution: {[f'{d:.3f}' for d in current_dist]}")
        print(f"  Target distribution:  {[f'{d:.3f}' for d in target_dist]}")
        
        def predict_with_scaling(scaling_factors):
            """Apply scaling factors to logits and predict."""
            # Convert probabilities back to logits (approximate)
            epsilon = 1e-8
            log_probs = np.log(np.clip(probs, epsilon, 1-epsilon))
            
            # Apply scaling factors
            scaled_logits = log_probs * scaling_factors[np.newaxis, :]
            
            # Convert back to probabilities and predict
            scaled_probs = np.exp(scaled_logits)
            scaled_probs = scaled_probs / scaled_probs.sum(axis=1, keepdims=True)
            
            return np.argmax(scaled_probs, axis=1)
        
        def objective(scaling_factors):
            """Objective function: MSE between predicted and target distribution."""
            try:
                predictions = predict_with_scaling(scaling_factors)
                pred_dist = np.array([(predictions == i).mean() for i in range(num_classes)])
                
                # Mean squared error between distributions
                mse = np.mean((pred_dist - target_dist) ** 2)
                
                # Add regularization to keep scaling factors reasonable
                reg = 0.01 * np.mean((scaling_factors - 1.0) ** 2)
                
                return mse + reg
            except:
                return 1e6  # Return high cost if numerical issues
        
        # Initialize scaling factors around 1.0
        initial_scaling = np.ones(num_classes)
        
        # Adjust initial values based on current vs target distribution
        for i in range(num_classes):
            if current_dist[i] > target_dist[i] and current_dist[i] > 0:
                # Class is over-predicted, decrease its scaling
                initial_scaling[i] = 0.8
            elif current_dist[i] < target_dist[i]:
                # Class is under-predicted, increase its scaling
                initial_scaling[i] = 1.2
        
        print(f"  Initial scaling:      {[f'{s:.3f}' for s in initial_scaling]}")
        
        # Bounds: scaling factors should be positive
        bounds = [(0.1, 3.0) for _ in range(num_classes)]
        
        # Try multiple optimization methods
        best_result = None
        best_cost = float('inf')
        
        for method in ['L-BFGS-B', 'SLSQP', 'Powell']:
            try:
                result = minimize(
                    objective,
                    initial_scaling,
                    method=method,
                    bounds=bounds if method != 'Powell' else None,
                    options={'maxiter': 200}
                )
                
                if result.success and result.fun < best_cost:
                    best_result = result
                    best_cost = result.fun
            except Exception as e:
                print(f"    Optimization with {method} failed: {e}")
                continue
        
        if best_result is None:
            print(f"    All optimization methods failed, using initial scaling")
            return initial_scaling
        
        final_scaling = best_result.x
        print(f"  Final scaling:        {[f'{s:.3f}' for s in final_scaling]}")
        print(f"  Optimization cost:    {best_result.fun:.6f}")
        
        # Test final distribution
        final_preds = predict_with_scaling(final_scaling)
        final_dist = np.array([(final_preds == i).mean() for i in range(num_classes)])
        print(f"  Final distribution:   {[f'{d:.3f}' for d in final_dist]}")
        
        return final_scaling
    
    print("\nOptimizing thresholds for left player...")
    dist_cal.thresholds_left = optimize_thresholds_for_side(all_probs_left, 'left')
    
    print("Optimizing thresholds for right player...")
    dist_cal.thresholds_right = optimize_thresholds_for_side(all_probs_right, 'right')
    
    # Evaluate final distribution
    print("\n=== Final Distribution After Calibration ===")
    final_pred_left = dist_cal.predict_with_thresholds(all_probs_left, 'left')
    final_pred_right = dist_cal.predict_with_thresholds(all_probs_right, 'right')
    
    print("Left player final distribution:")
    for class_id in range(num_classes):
        final_freq = (final_pred_left == class_id).mean()
        target_freq = target_distribution[class_id]
        print(f"  Class {class_id}: {final_freq:.3f} (target: {target_freq:.3f})")
    
    print("Right player final distribution:")
    for class_id in range(num_classes):
        final_freq = (final_pred_right == class_id).mean()
        target_freq = target_distribution[class_id]
        print(f"  Class {class_id}: {final_freq:.3f} (target: {target_freq:.3f})")
    
    print(f"\nLearned thresholds:")
    print(f"  Left:  {[f'{t:.3f}' for t in dist_cal.thresholds_left]}")
    print(f"  Right: {[f'{t:.3f}' for t in dist_cal.thresholds_right]}")
    
    return dist_cal


def learn_temperature_scaling_on_logits(
    logits_left: torch.Tensor,
    logits_right: torch.Tensor,
    labels_left: torch.Tensor,
    labels_right: torch.Tensor,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 0.01
) -> TemperatureScaling:
    """Learn temperature scaling parameters on pre-computed logits.
    
    Args:
        logits_left: Pre-computed left player logits (N, num_classes)
        logits_right: Pre-computed right player logits (N, num_classes)
        labels_left: True labels for left player (N,)
        labels_right: True labels for right player (N,)
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate for temperature optimization
        
    Returns:
        Trained TemperatureScaling module
    """
    temperature_scaling = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temperature_scaling.temperature_left, temperature_scaling.temperature_right], lr=lr, max_iter=max_iter)
    
    # Move tensors to device
    logits_left = logits_left.to(device)
    logits_right = logits_right.to(device)
    labels_left = labels_left.to(device)
    labels_right = labels_right.to(device)
    
    print("Optimizing temperature parameters on ensemble logits...")
    
    def eval_loss():
        # Apply temperature scaling
        scaled_logits_left, scaled_logits_right = temperature_scaling(logits_left, logits_right)
        
        # Calculate NLL loss
        loss_left = F.cross_entropy(scaled_logits_left, labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, labels_right)
        total_loss = (loss_left + loss_right) / 2
        
        return total_loss
    
    # Initial metrics
    with torch.no_grad():
        initial_loss = eval_loss().item()
        
        # Calculate initial ECE
        probs_left = F.softmax(logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(logits_right, dim=1).cpu().numpy()
        labels_left_np = labels_left.cpu().numpy()
        labels_right_np = labels_right.cpu().numpy()
        
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
        scaled_logits_left, scaled_logits_right = temperature_scaling(logits_left, logits_right)
        probs_left = F.softmax(scaled_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(scaled_logits_right, dim=1).cpu().numpy()
        
        ece_left_after = expected_calibration_error(labels_left_np, probs_left)
        ece_right_after = expected_calibration_error(labels_right_np, probs_right)
        ece_after = (ece_left_after + ece_right_after) / 2
    
    temps = temperature_scaling.get_temperatures()
    print(f"After calibration - Loss: {final_loss:.4f}, ECE: {ece_after:.4f}")
    print(f"Learned temperatures - Left: {temps['temperature_left']:.4f}, Right: {temps['temperature_right']:.4f}")
    
    return temperature_scaling


def learn_temperature_scaling_f1_on_logits(
    logits_left: torch.Tensor,
    logits_right: torch.Tensor,
    labels_left: torch.Tensor,
    labels_right: torch.Tensor,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 0.01
) -> TemperatureScaling:
    """Learn temperature scaling parameters optimizing for F1 score on pre-computed logits.
    
    Args:
        logits_left: Pre-computed left player logits (N, num_classes)
        logits_right: Pre-computed right player logits (N, num_classes)
        labels_left: True labels for left player (N,)
        labels_right: True labels for right player (N,)
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate for temperature optimization
        
    Returns:
        Trained TemperatureScaling module
    """
    temperature_scaling = TemperatureScaling().to(device)
    optimizer = torch.optim.Adam([temperature_scaling.temperature_left, temperature_scaling.temperature_right], lr=lr)
    
    # Move tensors to device
    logits_left = logits_left.to(device)
    logits_right = logits_right.to(device)
    labels_left = labels_left.to(device)
    labels_right = labels_right.to(device)
    
    print("Optimizing temperature parameters for F1 score on ensemble logits...")
    
    def calculate_f1_score():
        # Apply temperature scaling
        scaled_logits_left, scaled_logits_right = temperature_scaling(logits_left, logits_right)
        
        # Get predictions
        pred_left = scaled_logits_left.argmax(dim=1).cpu().numpy()
        pred_right = scaled_logits_right.argmax(dim=1).cpu().numpy()
        
        # Convert labels to numpy
        labels_left_np = labels_left.cpu().numpy()
        labels_right_np = labels_right.cpu().numpy()
        
        # Calculate F1 scores
        f1_left = f1_score(labels_left_np, pred_left, average='macro', zero_division=0)
        f1_right = f1_score(labels_right_np, pred_right, average='macro', zero_division=0)
        f1_avg = (f1_left + f1_right) / 2
        
        return f1_avg, f1_left, f1_right
    
    # Initial metrics
    with torch.no_grad():
        f1_before, f1_left_before, f1_right_before = calculate_f1_score()
    
    print(f"Before calibration - F1: {f1_before:.4f} (L:{f1_left_before:.4f}, R:{f1_right_before:.4f})")
    
    # Optimize temperature to maximize F1 score
    best_f1 = f1_before
    best_temperatures = (temperature_scaling.temperature_left.item(), temperature_scaling.temperature_right.item())
    
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Use cross-entropy as proxy for F1 optimization with regularization
        scaled_logits_left, scaled_logits_right = temperature_scaling(logits_left, logits_right)
        
        loss_left = F.cross_entropy(scaled_logits_left, labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, labels_right)
        loss = (loss_left + loss_right) / 2
        
        # Add regularization to prevent extreme temperature values
        reg_loss = 0.001 * (
            torch.abs(temperature_scaling.temperature_left - 1.0) + 
            torch.abs(temperature_scaling.temperature_right - 1.0)
        )
        
        total_loss = loss + reg_loss
        total_loss.backward()
        optimizer.step()
        
        # Check F1 score every few iterations
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                f1_current, f1_left_current, f1_right_current = calculate_f1_score()
                if f1_current > best_f1:
                    best_f1 = f1_current
                    best_temperatures = (temperature_scaling.temperature_left.item(), temperature_scaling.temperature_right.item())
                
                print(f"Iteration {i+1}/{max_iter}, F1: {f1_current:.4f} (L:{f1_left_current:.4f}, R:{f1_right_current:.4f})")
    
    # Set best temperatures
    temperature_scaling.temperature_left.data = torch.tensor([best_temperatures[0]], device=device)
    temperature_scaling.temperature_right.data = torch.tensor([best_temperatures[1]], device=device)
    
    # Final metrics
    with torch.no_grad():
        f1_after, f1_left_after, f1_right_after = calculate_f1_score()
    
    temps = temperature_scaling.get_temperatures()
    print(f"After calibration - F1: {f1_after:.4f} (L:{f1_left_after:.4f}, R:{f1_right_after:.4f})")
    print(f"Learned temperatures - Left: {temps['temperature_left']:.4f}, Right: {temps['temperature_right']:.4f}")
    
    return temperature_scaling


def learn_vector_scaling_on_logits(
    logits_left: torch.Tensor,
    logits_right: torch.Tensor,
    labels_left: torch.Tensor,
    labels_right: torch.Tensor,
    num_classes: int,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 0.01
) -> VectorScaling:
    """Learn vector scaling parameters on pre-computed logits.
    
    Args:
        logits_left: Pre-computed left player logits (N, num_classes)
        logits_right: Pre-computed right player logits (N, num_classes)
        labels_left: True labels for left player (N,)
        labels_right: True labels for right player (N,)
        num_classes: Number of classes
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate for optimization
        
    Returns:
        Trained VectorScaling module
    """
    vector_scaling = VectorScaling(num_classes).to(device)
    optimizer = torch.optim.Adam([
        vector_scaling.w_left, vector_scaling.b_left,
        vector_scaling.w_right, vector_scaling.b_right
    ], lr=lr)
    
    # Move tensors to device
    logits_left = logits_left.to(device)
    logits_right = logits_right.to(device)
    labels_left = labels_left.to(device)
    labels_right = labels_right.to(device)
    
    print("Optimizing vector scaling parameters on ensemble logits...")
    
    # Initial metrics
    with torch.no_grad():
        probs_left = F.softmax(logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(logits_right, dim=1).cpu().numpy()
        labels_left_np = labels_left.cpu().numpy()
        labels_right_np = labels_right.cpu().numpy()
        
        ece_left_before = expected_calibration_error(labels_left_np, probs_left)
        ece_right_before = expected_calibration_error(labels_right_np, probs_right)
        ece_before = (ece_left_before + ece_right_before) / 2
        
        initial_loss_left = F.cross_entropy(logits_left, labels_left).item()
        initial_loss_right = F.cross_entropy(logits_right, labels_right).item()
        initial_loss = (initial_loss_left + initial_loss_right) / 2
    
    print(f"Before calibration - Loss: {initial_loss:.4f}, ECE: {ece_before:.4f}")
    
    # Optimize parameters
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Apply vector scaling
        scaled_logits_left, scaled_logits_right = vector_scaling(logits_left, logits_right)
        
        # Calculate NLL loss
        loss_left = F.cross_entropy(scaled_logits_left, labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, labels_right)
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
        scaled_logits_left, scaled_logits_right = vector_scaling(logits_left, logits_right)
        
        probs_left = F.softmax(scaled_logits_left, dim=1).cpu().numpy()
        probs_right = F.softmax(scaled_logits_right, dim=1).cpu().numpy()
        
        ece_left_after = expected_calibration_error(labels_left_np, probs_left)
        ece_right_after = expected_calibration_error(labels_right_np, probs_right)
        ece_after = (ece_left_after + ece_right_after) / 2
        
        final_loss_left = F.cross_entropy(scaled_logits_left, labels_left).item()
        final_loss_right = F.cross_entropy(scaled_logits_right, labels_right).item()
        final_loss = (final_loss_left + final_loss_right) / 2
    
    print(f"After calibration - Loss: {final_loss:.4f}, ECE: {ece_after:.4f}")
    
    return vector_scaling


def learn_vector_scaling_f1_on_logits(
    logits_left: torch.Tensor,
    logits_right: torch.Tensor,
    labels_left: torch.Tensor,
    labels_right: torch.Tensor,
    num_classes: int,
    device: torch.device,
    max_iter: int = 200,
    lr: float = 0.01
) -> VectorScaling:
    """Learn vector scaling parameters optimizing for F1 score on pre-computed logits.
    
    Args:
        logits_left: Pre-computed left player logits (N, num_classes)
        logits_right: Pre-computed right player logits (N, num_classes)
        labels_left: True labels for left player (N,)
        labels_right: True labels for right player (N,)
        num_classes: Number of classes
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate for optimization
        
    Returns:
        Trained VectorScaling module
    """
    vector_scaling = VectorScaling(num_classes).to(device)
    optimizer = torch.optim.Adam([
        vector_scaling.w_left, vector_scaling.b_left,
        vector_scaling.w_right, vector_scaling.b_right
    ], lr=lr)
    
    # Move tensors to device
    logits_left = logits_left.to(device)
    logits_right = logits_right.to(device)
    labels_left = labels_left.to(device)
    labels_right = labels_right.to(device)
    
    print("Optimizing vector scaling parameters for F1 score on ensemble logits...")
    
    def calculate_f1_score():
        # Apply vector scaling
        scaled_logits_left, scaled_logits_right = vector_scaling(logits_left, logits_right)
        
        # Get predictions
        pred_left = scaled_logits_left.argmax(dim=1).cpu().numpy()
        pred_right = scaled_logits_right.argmax(dim=1).cpu().numpy()
        
        # Convert labels to numpy
        labels_left_np = labels_left.cpu().numpy()
        labels_right_np = labels_right.cpu().numpy()
        
        # Calculate F1 scores
        f1_left = f1_score(labels_left_np, pred_left, average='macro', zero_division=0)
        f1_right = f1_score(labels_right_np, pred_right, average='macro', zero_division=0)
        f1_avg = (f1_left + f1_right) / 2
        
        return f1_avg, f1_left, f1_right
    
    # Initial metrics
    with torch.no_grad():
        f1_before, f1_left_before, f1_right_before = calculate_f1_score()
    
    print(f"Before calibration - F1: {f1_before:.4f} (L:{f1_left_before:.4f}, R:{f1_right_before:.4f})")
    
    # Track best F1 score and parameters
    best_f1 = f1_before
    best_state_dict = vector_scaling.state_dict().copy()
    
    # Optimize parameters
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Apply vector scaling
        scaled_logits_left, scaled_logits_right = vector_scaling(logits_left, logits_right)
        
        # Calculate cross-entropy loss as proxy for F1 optimization
        loss_left = F.cross_entropy(scaled_logits_left, labels_left)
        loss_right = F.cross_entropy(scaled_logits_right, labels_right)
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
        
        # Check F1 score every few iterations and save best
        if (i + 1) % 10 == 0:
            with torch.no_grad():
                f1_current, f1_left_current, f1_right_current = calculate_f1_score()
                if f1_current > best_f1:
                    best_f1 = f1_current
                    best_state_dict = vector_scaling.state_dict().copy()
                
                print(f"Iteration {i+1}/{max_iter}, F1: {f1_current:.4f} (L:{f1_left_current:.4f}, R:{f1_right_current:.4f})")
    
    # Load best parameters
    vector_scaling.load_state_dict(best_state_dict)
    
    # Final metrics
    with torch.no_grad():
        f1_after, f1_left_after, f1_right_after = calculate_f1_score()
    
    print(f"After calibration - F1: {f1_after:.4f} (L:{f1_left_after:.4f}, R:{f1_right_after:.4f})")
    
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


