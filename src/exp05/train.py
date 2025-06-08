"""Training script for exp05"""

import argparse
import warnings
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
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
        
        print(f"\nUsing fold {self.config.fold + 1}/{self.config.n_folds}")
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
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
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
            x, y_left, y_right = batch
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
    
    # Save predictions
    output_path = config.output_dir / f"predictions_fold_{config.fold}.csv"
    save_predictions(result_df, output_path, config)
    
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


def main():
    parser = argparse.ArgumentParser(description="Train Epee exp05 model")
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
    
    # Save config
    config.save(config.output_dir / "config.json")
    
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