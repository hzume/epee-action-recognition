"""Training script for exp07 with frame difference features"""

import argparse
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Optional
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import EpeeDataset
from .model import ImprovedLSTMModel, LitModel
from .utils import (
    Config,
    evaluate_predictions,
    print_classification_report,
    save_predictions,
    save_normalization_stats,
    load_normalization_stats
)
from .calibration import (
    TemperatureScaling,
    VectorScaling,
    load_temperature_scaling,
    load_vector_scaling,
    apply_calibration_to_ensemble_logits,
    sample_predictions_with_temperature
)

warnings.filterwarnings("ignore")


class EpeeDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for Epee dataset"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Load data
        self.df = pd.read_csv(config.data_dir / "pose_preds.csv")
        
        # Split prediction videos
        self.pred_df = self.df[self.df["video_filename"].isin(config.predict_videos)]
        self.train_val_df = self.df[~self.df["video_filename"].isin(config.predict_videos)]
        
        # Cross-validation split
        self._setup_cv_split()
        
    def _setup_cv_split(self):
        """Setup cross-validation splits"""
        skf = StratifiedGroupKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed
        )
        
        # Use left_action for stratification (fill NaN with 'none')
        stratify_labels = self.train_val_df["left_action"].fillna("none")
        groups = self.train_val_df["video_filename"]
        
        # Get all splits
        splits = list(skf.split(self.train_val_df, stratify_labels, groups))
        
        # Select the specified fold
        train_idx, valid_idx = splits[self.config.fold]
        
        self.train_df = self.train_val_df.iloc[train_idx]
        self.valid_df = self.train_val_df.iloc[valid_idx]
        
        if self.config.debug:
            train_first_video = self.train_df["video_filename"].unique()[0]
            self.train_df = self.train_df[self.train_df["video_filename"] == train_first_video]
            valid_first_video = self.valid_df["video_filename"].unique()[0]
            self.valid_df = self.valid_df[self.valid_df["video_filename"] == valid_first_video]
            pred_first_video = self.pred_df["video_filename"].unique()[0]
            self.pred_df = self.pred_df[self.pred_df["video_filename"] == pred_first_video]


        print(f"\nUsing fold {self.config.fold + 1}/{self.config.n_folds}")
        print(f"Train samples: {len(self.train_df)}")
        print(f"Valid samples: {len(self.valid_df)}")
        print(f"Train videos: {sorted(self.train_df['video_filename'].unique())}")
        print(f"Valid videos: {sorted(self.valid_df['video_filename'].unique())}")
        print(f"Pred videos: {sorted(self.pred_df['video_filename'].unique())}")
    
    def setup(self, stage: str):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            # Create training dataset first to compute normalization stats
            self.train_dataset = EpeeDataset(
                self.train_df,
                self.config,
                sample_balanced=True,
                augment=True
            )
            
            # Use training normalization stats for validation dataset
            self.valid_dataset = EpeeDataset(
                self.valid_df,
                self.config,
                sample_balanced=False,
                augment=False,
                normalization_stats=self.train_dataset.normalization_stats
            )
            
            # Store feature columns for model initialization
            self.num_features = len(self.train_dataset.feature_cols)
            
        if stage == "predict" or stage is None:
            # For prediction, we need to use the training normalization stats
            # Create a temporary training dataset if not already created
            if not hasattr(self, 'train_dataset'):
                self.train_dataset = EpeeDataset(
                    self.train_df,
                    self.config,
                    sample_balanced=False,
                    augment=False
                )
            
            self.pred_dataset = EpeeDataset(
                self.pred_df,
                self.config,
                sample_balanced=False,
                augment=False,
                normalization_stats=self.train_dataset.normalization_stats
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Important: keep order for temporal consistency
            num_workers=4,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Important: keep order for temporal consistency
            num_workers=4,
            pin_memory=True
        )


def train_model(config: Config):
    """Train the model"""
    # Set random seeds
    L.seed_everything(config.seed)
    
    # Setup data module
    dm = EpeeDataModule(config)
    dm.setup("fit")
    
    # Calculate training steps
    num_training_steps = len(dm.train_dataloader()) * config.epochs
    
    # Initialize model
    model = ImprovedLSTMModel(
        num_features=dm.num_features,
        config=config
    )
    
    # Lightning module
    lit_model = LitModel(
        model=model,
        config=config,
        num_training_steps=num_training_steps
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.output_dir / f"fold_{config.fold}",
            filename="{epoch:02d}-{val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/f1_macro",
            mode="max",
            patience=10,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.output_dir,
        name=f"fold_{config.fold}"
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5
    )
    
    # Train
    trainer.fit(lit_model, dm)
    
    # Save normalization statistics
    norm_stats_path = config.output_dir / f"fold_{config.fold}" / "normalization_stats.json"
    save_normalization_stats(dm.train_dataset.normalization_stats, norm_stats_path)
    
    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nLoading best model from: {best_model_path}")
    lit_model = LitModel.load_from_checkpoint(
        best_model_path,
        model=model,
        config=config
    )
    
    return trainer, lit_model, dm


def evaluate_model(trainer, lit_model, dm, config):
    """Evaluate the model on validation and test sets"""
    # Validation set evaluation
    print("\n" + "="*50)
    print("VALIDATION SET EVALUATION")
    print("="*50)
    
    lit_model.eval()
    all_preds_left = []
    all_preds_right = []
    all_true_left = []
    all_true_right = []
    
    with torch.no_grad():
        for batch in dm.val_dataloader():
            x, y_left, y_right, info = batch
            y_left_hat, y_right_hat = lit_model(x.to(lit_model.device))
            
            all_preds_left.append(y_left_hat.argmax(dim=1).cpu().numpy())
            all_preds_right.append(y_right_hat.argmax(dim=1).cpu().numpy())
            all_true_left.append(y_left.argmax(dim=1).cpu().numpy())
            all_true_right.append(y_right.argmax(dim=1).cpu().numpy())
    
    # Concatenate all predictions
    pred_left = np.concatenate(all_preds_left)
    pred_right = np.concatenate(all_preds_right)
    true_left = np.concatenate(all_true_left)
    true_right = np.concatenate(all_true_right)
    
    # Get class names
    class_names = ["none"] + list(config.metadata["action_to_id"].keys())
    
    # Evaluate
    metrics_left = evaluate_predictions(true_left, pred_left, class_names, "left_")
    metrics_right = evaluate_predictions(true_right, pred_right, class_names, "right_")
    
    # Combined metrics
    metrics = {
        "overall_accuracy": (metrics_left["left_accuracy"] + metrics_right["right_accuracy"]) / 2,
        "overall_accuracy_ignore_none": (
            metrics_left["left_accuracy_ignore_none"] + 
            metrics_right["right_accuracy_ignore_none"]
        ) / 2,
        "overall_f1_macro": (metrics_left["left_f1_macro"] + metrics_right["right_f1_macro"]) / 2,
    }
    metrics.update(metrics_left)
    metrics.update(metrics_right)
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Accuracy (ignore none): {metrics['overall_accuracy_ignore_none']:.4f}")
    print(f"  F1 Macro: {metrics['overall_f1_macro']:.4f}")
    
    # Print classification reports
    print_classification_report(true_left, pred_left, class_names, "Left Player")
    print_classification_report(true_right, pred_right, class_names, "Right Player")
    
    return metrics


def calculate_label_switches(predictions):
    """Calculate the number of label switches in predictions"""
    switches = 0
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i-1]:
            switches += 1
    return switches


def generate_predictions(trainer, lit_model, dm, config):
    """Generate predictions for test videos"""
    print("\n" + "="*50)
    print("GENERATING PREDICTIONS")
    print("="*50)
    
    # Setup prediction dataset
    dm.setup("predict")
    
    # Run predictions
    predictions = trainer.predict(lit_model, dm)
    
    # Combine predictions
    all_logits_left = torch.cat([p[0] for p in predictions], dim=0)
    all_logits_right = torch.cat([p[1] for p in predictions], dim=0)
    
    pred_ids_left = all_logits_left.argmax(dim=1).cpu().numpy()
    pred_ids_right = all_logits_right.argmax(dim=1).cpu().numpy()
    
    # Create result dataframe
    result_data = []
    for i in range(len(dm.pred_dataset)):
        item = dm.pred_dataset.data[i]
        if not item["switched"]:  # Only include non-switched predictions
            result_data.append({
                "video_filename": item["video_filename"],
                "frame_filename": item["frame_filename"],
                "frame_idx": item["frame_idx"],
                "left_pred_action_id": pred_ids_left[i],
                "right_pred_action_id": pred_ids_right[i],
            })
    
    result_df = pd.DataFrame(result_data)
    
    # Calculate label switches per video
    print("\nLabel switching analysis:")
    for video in result_df["video_filename"].unique():
        video_df = result_df[result_df["video_filename"] == video].sort_values("frame_idx")
        left_switches = calculate_label_switches(video_df["left_pred_action_id"].values)
        right_switches = calculate_label_switches(video_df["right_pred_action_id"].values)
        total_frames = len(video_df)
        
        print(f"{video}:")
        print(f"  Left player: {left_switches} switches ({left_switches/total_frames:.2%})")
        print(f"  Right player: {right_switches} switches ({right_switches/total_frames:.2%})")
    
    # Save predictions
    output_path = config.output_dir / f"predictions_fold_{config.fold}.csv"
    save_predictions(result_df, output_path, config)
    
    return result_df


def generate_ensemble_predictions(
    config: Config, 
    use_calibration: bool = True, 
    calibration_method: str = "temperature", 
    calibration_objective: str = "ece", 
    sampling_temperature: float = 1.0,
    random_seed: Optional[int] = None
):
    """Generate ensemble predictions using new approach: logits averaging → calibration → sampling
    
    Args:
        config: Configuration object
        use_calibration: Whether to use calibration if available
        calibration_method: Calibration method to use ('temperature' or 'vector')
        calibration_objective: Calibration objective used ('ece' or 'f1')
        sampling_temperature: Temperature for probabilistic sampling (1.0 = standard sampling)
        random_seed: Random seed for reproducible sampling
    """
    print("\n" + "="*60)
    print("GENERATING ENSEMBLE PREDICTIONS")
    print("NEW APPROACH: Logits Averaging → Calibration → Probabilistic Sampling")
    if use_calibration:
        print(f"Calibration: {calibration_method} scaling ({calibration_objective} optimized)")
    else:
        print("Calibration: None (raw logits)")
    print(f"Sampling: Temperature = {sampling_temperature}")
    if random_seed is not None:
        print(f"Random seed: {random_seed}")
    print("="*60)
    
    # Collect all fold checkpoints
    fold_checkpoints = []
    for fold in range(config.n_folds):
        fold_dir = config.output_dir / f"fold_{fold}"
        # Find best checkpoint
        ckpt_files = list(fold_dir.glob("*.ckpt"))
        if not ckpt_files:
            print(f"Warning: No checkpoint found for fold {fold}")
            continue
        
        # Find best checkpoint by looking for one with highest metric in filename
        best_ckpt = None
        best_metric = -1
        for ckpt_file in ckpt_files:
            if "last.ckpt" in ckpt_file.name:
                continue
            try:
                # Extract metric from filename (format: epoch-metric.ckpt)
                parts = ckpt_file.stem.split('-')
                if len(parts) >= 2:
                    metric_str = parts[1]
                    metric_val = float(metric_str)
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_ckpt = ckpt_file
            except (ValueError, IndexError):
                continue
        
        if best_ckpt is None and ckpt_files:
            best_ckpt = ckpt_files[0]  # Fallback to first available
        fold_checkpoints.append((fold, best_ckpt))
    
    print(f"Found {len(fold_checkpoints)} fold checkpoints")
    
    # Initialize data module for prediction (no data limiting for prediction)
    dm = EpeeDataModule(config)
    dm.setup("predict")
    
    # Collect predictions from all folds
    all_fold_logits_left = []
    all_fold_logits_right = []
    
    for fold_idx, ckpt_path in fold_checkpoints:
        print(f"\nProcessing fold {fold_idx}...")
        
        # Load normalization stats for this fold
        norm_stats_path = config.output_dir / f"fold_{fold_idx}" / "normalization_stats.json"
        if norm_stats_path.exists():
            norm_stats = load_normalization_stats(norm_stats_path)
        else:
            print(f"Warning: No normalization stats found for fold {fold_idx}")
            continue
        
        # Create dataset with fold-specific normalization
        pred_dataset = EpeeDataset(
            dm.pred_df,
            config,
            sample_balanced=False,
            augment=False,
            normalization_stats=norm_stats
        )
        
        # Create dataloader
        pred_loader = DataLoader(
            pred_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Load model
        model = ImprovedLSTMModel(
            num_features=len(pred_dataset.feature_cols),
            config=config
        )
        lit_model = LitModel.load_from_checkpoint(
            ckpt_path,
            model=model,
            config=config
        )
        lit_model.eval()
        
        # Determine device
        device = next(lit_model.parameters()).device
        
        # NEW APPROACH: No per-fold calibration, collect raw logits only
        print(f"  Collecting raw logits from fold {fold_idx} (calibration will be applied after ensemble averaging)")
        
        # Collect predictions for this fold
        fold_logits_left = []
        fold_logits_right = []
        
        with torch.no_grad():
            for batch in pred_loader:
                x, y_left, y_right, info = batch
                
                # Normal prediction
                y_left_hat, y_right_hat = lit_model(x.to(device))
                
                # TTA: Use switched data (already in dataset)
                # Average normal and switched predictions
                batch_logits_left = y_left_hat.cpu()
                batch_logits_right = y_right_hat.cpu()
                
                fold_logits_left.append(batch_logits_left)
                fold_logits_right.append(batch_logits_right)
        
        # Combine batch predictions
        fold_logits_left = torch.cat(fold_logits_left, dim=0)
        fold_logits_right = torch.cat(fold_logits_right, dim=0)
        
        # Separate normal and switched predictions for TTA
        # Dataset contains both switched=False and switched=True data interleaved
        normal_indices = []
        switched_indices = []
        
        for i, item in enumerate(pred_dataset.data):
            if item["switched"]:
                switched_indices.append(i)
            else:
                normal_indices.append(i)
        
        normal_left = fold_logits_left[normal_indices]
        normal_right = fold_logits_right[normal_indices]
        switched_left = fold_logits_left[switched_indices]
        switched_right = fold_logits_right[switched_indices]
        
        # TTA: Average normal and switched (swapped back for switched data)
        tta_left = (normal_left + switched_right) / 2
        tta_right = (normal_right + switched_left) / 2
        
        all_fold_logits_left.append(tta_left)
        all_fold_logits_right.append(tta_right)
    
    # Average across all folds
    ensemble_logits_left = torch.stack(all_fold_logits_left).mean(dim=0)
    ensemble_logits_right = torch.stack(all_fold_logits_right).mean(dim=0)
    
    print(f"\nEnsemble logits shape: Left={ensemble_logits_left.shape}, Right={ensemble_logits_right.shape}")
    
    # Load and apply calibration to ensemble-averaged logits
    calibration_module = None
    if use_calibration:
        suffix = f"_{calibration_objective}" if calibration_objective != "ece" else ""
        
        # We'll use the calibration from fold 0 as representative
        # (In practice, you might want to average calibration parameters or use a specific fold)
        fold_dir = config.output_dir / "fold_0"
        
        if calibration_method == "temperature":
            calibration_path = fold_dir / f"temperature_scaling{suffix}.pth"
            if calibration_path.exists():
                print(f"Loading {calibration_method} calibration ({calibration_objective}) for ensemble")
                calibration_module = TemperatureScaling()
                calibration_module.load_state_dict(torch.load(calibration_path, map_location='cpu'))
                calibration_module.eval()
            else:
                print(f"No {calibration_method} calibration found, using raw logits")
                
        elif calibration_method == "vector":
            calibration_path = fold_dir / f"vector_scaling{suffix}.pth"
            if calibration_path.exists():
                print(f"Loading {calibration_method} calibration ({calibration_objective}) for ensemble")
                calibration_module = VectorScaling(config.num_classes)
                calibration_module.load_state_dict(torch.load(calibration_path, map_location='cpu'))
                calibration_module.eval()
            else:
                print(f"No {calibration_method} calibration found, using raw logits")
    else:
        print("No calibration requested, using raw logits")
    
    # Apply calibration to ensemble logits
    calibrated_logits_left, calibrated_logits_right = apply_calibration_to_ensemble_logits(
        ensemble_logits_left, ensemble_logits_right, calibration_module
    )
    
    # Generate predictions using probabilistic sampling
    print(f"Generating predictions using temperature sampling (temperature={sampling_temperature})")
    pred_ids_left = sample_predictions_with_temperature(
        calibrated_logits_left, temperature=sampling_temperature, random_seed=random_seed
    ).cpu().numpy()
    pred_ids_right = sample_predictions_with_temperature(
        calibrated_logits_right, temperature=sampling_temperature, random_seed=random_seed
    ).cpu().numpy()
    
    # Create result dataframe (only non-switched data)
    # Recreate a basic dataset just for metadata
    basic_dataset = EpeeDataset(
        dm.pred_df,
        config,
        sample_balanced=False,
        augment=False
    )
    
    result_data = []
    normal_idx = 0
    for i, item in enumerate(basic_dataset.data):
        if not item["switched"]:
            result_data.append({
                "video_filename": item["video_filename"],
                "frame_filename": item["frame_filename"],
                "frame_idx": item["frame_idx"],
                "left_pred_action_id": pred_ids_left[normal_idx],
                "right_pred_action_id": pred_ids_right[normal_idx],
            })
            normal_idx += 1
    
    result_df = pd.DataFrame(result_data)
    
    # Calculate label switches per video
    print("\nLabel switching analysis (Ensemble):")
    for video in result_df["video_filename"].unique():
        video_df = result_df[result_df["video_filename"] == video].sort_values("frame_idx")
        left_switches = calculate_label_switches(video_df["left_pred_action_id"].values)
        right_switches = calculate_label_switches(video_df["right_pred_action_id"].values)
        total_frames = len(video_df)
        
        print(f"{video}:")
        print(f"  Left player: {left_switches} switches ({left_switches/total_frames:.2%})")
        print(f"  Right player: {right_switches} switches ({right_switches/total_frames:.2%})")
    
    # Save ensemble predictions with new naming scheme
    cal_suffix = f"_{calibration_method}_{calibration_objective}" if use_calibration else "_uncalibrated"
    temp_suffix = f"_temp{sampling_temperature}" if sampling_temperature != 1.0 else ""
    seed_suffix = f"_seed{random_seed}" if random_seed is not None else ""
    output_path = config.output_dir / f"predictions_ensemble_sampling{cal_suffix}{temp_suffix}{seed_suffix}.csv"
    save_predictions(result_df, output_path, config)
    print(f"\nSampling-based ensemble predictions saved to: {output_path}")
    
    return result_df


def run_cross_validation(config: Config):
    """Run full cross-validation"""
    all_metrics = []
    
    for fold in range(config.n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{config.n_folds}")
        print(f"{'='*60}")
        
        # Update config for this fold
        config.fold = fold
        
        # Train model
        trainer, lit_model, dm = train_model(config)
        
        # Evaluate
        metrics = evaluate_model(trainer, lit_model, dm, config)
        metrics["fold"] = fold
        all_metrics.append(metrics)
        
        # Generate predictions
        generate_predictions(trainer, lit_model, dm, config)
    
    # Save CV results
    cv_results_df = pd.DataFrame(all_metrics)
    cv_results_df.to_csv(config.output_dir / "cv_results.csv", index=False)
    
    # Print CV summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    for metric in ["overall_accuracy", "overall_accuracy_ignore_none", "overall_f1_macro"]:
        mean_val = cv_results_df[metric].mean()
        std_val = cv_results_df[metric].std()
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save as JSON too
    cv_summary = {
        "mean": cv_results_df.mean().to_dict(),
        "std": cv_results_df.std().to_dict(),
        "per_fold": all_metrics
    }
    
    with open(config.output_dir / "cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=4)
    
    
    # Generate ensemble predictions with calibration if available
    generate_ensemble_predictions(config, use_calibration=True)


def main():
    parser = argparse.ArgumentParser(description="Train Epee exp07 model with frame difference features")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to run (0-3). If not specified, runs all folds"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: reduced data and faster training"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    config.debug = args.debug
    
    # Debug mode adjustments
    if args.debug:
        config.epochs = min(config.epochs, 2)
        config.n_folds = min(config.n_folds, 2)
        print("Debug mode enabled: reducing epochs, batch size, and folds")
    
    
    
    # Save config
    config.save(config.output_dir / "config.json")
    
    print(f"Configuration:")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    
    # Run training
    if args.fold is not None:
        # Run single fold
        config.fold = args.fold
        trainer, lit_model, dm = train_model(config)
        metrics = evaluate_model(trainer, lit_model, dm, config)
        generate_predictions(trainer, lit_model, dm, config)
    else:
        # Run full cross-validation
        run_cross_validation(config)


if __name__ == "__main__":
    main()