"""Training script for exp08 - 3D pose-based action recognition"""

import argparse
import warnings
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np

from .dataset import create_3d_pose_datasets
from .model import ThreeDLitModel
from .utils import (
    Config3D,
    evaluate_3d_predictions,
    print_3d_classification_report,
    save_3d_predictions
)

warnings.filterwarnings("ignore")


class ThreeDPoseDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for 3D pose dataset"""
    
    def __init__(self, config: Config3D):
        super().__init__()
        self.config = config
        
    def setup(self, stage: str):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset, self.test_dataset = create_3d_pose_datasets(self.config)
            
            # Store for access
            self.num_features = self.train_dataset[0][0].shape[-1] if len(self.train_dataset) > 0 else 0
            
        if stage == "predict" or stage is None:
            if not hasattr(self, 'test_dataset'):
                _, _, self.test_dataset = create_3d_pose_datasets(self.config)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )


def train_3d_model(config: Config3D, fold: int = 0):
    """Train 3D pose model"""
    # Set random seeds
    L.seed_everything(config.cross_validation["seed"])
    
    # Setup data module
    dm = ThreeDPoseDataModule(config)
    dm.setup("fit")
    
    if len(dm.train_dataset) == 0:
        print("No training data found. Please run 3D pose preprocessing first.")
        return None, None, None
    
    # Calculate training steps
    num_training_steps = len(dm.train_dataloader()) * config.training["epochs"]
    
    # Initialize Lightning model
    lit_model = ThreeDLitModel(
        config=config,
        num_training_steps=num_training_steps
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.output_dir / f"fold_{fold}",
            filename="{epoch:02d}-{val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/f1_macro",
            mode="max",
            patience=15,
            verbose=True
        ),
        LearningRateMonitor(logging_interval="step")
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.output_dir,
        name=f"fold_{fold}"
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=config.training["epochs"],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision=16,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5
    )
    
    print(f"\nTraining 3D pose model (fold {fold})")
    print(f"Train samples: {len(dm.train_dataset)}")
    print(f"Val samples: {len(dm.val_dataset)}")
    print(f"Test samples: {len(dm.test_dataset)}")
    
    # Train
    trainer.fit(lit_model, dm)
    
    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nLoading best model from: {best_model_path}")
    lit_model = ThreeDLitModel.load_from_checkpoint(
        best_model_path,
        config=config
    )
    
    return trainer, lit_model, dm


def evaluate_3d_model(trainer, lit_model, dm, config):
    """Evaluate 3D pose model"""
    print("\n" + "="*50)
    print("3D POSE MODEL EVALUATION")
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
    
    # Concatenate predictions
    pred_left = np.concatenate(all_preds_left)
    pred_right = np.concatenate(all_preds_right)
    true_left = np.concatenate(all_true_left)
    true_right = np.concatenate(all_true_right)
    
    # Class names
    class_names = ["none"] + list(config.metadata["action_to_id"].keys())
    
    # Evaluate
    metrics_left = evaluate_3d_predictions(true_left, pred_left, class_names, "left_")
    metrics_right = evaluate_3d_predictions(true_right, pred_right, class_names, "right_")
    
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
    print(f"\n3D Pose Model Metrics:")
    print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Accuracy (ignore none): {metrics['overall_accuracy_ignore_none']:.4f}")
    print(f"  F1 Macro: {metrics['overall_f1_macro']:.4f}")
    
    # Print classification reports
    print_3d_classification_report(true_left, pred_left, class_names, "Left Player (3D)")
    print_3d_classification_report(true_right, pred_right, class_names, "Right Player (3D)")
    
    return metrics


def generate_3d_predictions(trainer, lit_model, dm, config, fold: int = 0):
    """Generate predictions for test videos using 3D pose model"""
    print("\n" + "="*50)
    print("GENERATING 3D POSE PREDICTIONS")
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
    for i, item in enumerate(dm.test_dataset.data):
        result_data.append({
            "video_filename": item["video_filename"],
            "frame_idx": item["frame_idx"],
            "instance_id": item["instance_id"],
            "left_pred_action_id": pred_ids_left[i],
            "right_pred_action_id": pred_ids_right[i],
        })
    
    result_df = pd.DataFrame(result_data)
    
    # Calculate prediction statistics
    print(f"\n3D Pose Prediction Statistics:")
    print(f"  Total predictions: {len(result_df)}")
    print(f"  Videos: {sorted(result_df['video_filename'].unique())}")
    
    # Action distribution
    id_to_action = {v: k for k, v in config.metadata["action_to_id"].items()}
    id_to_action[0] = "none"
    
    for side in ["left", "right"]:
        print(f"\n  {side.capitalize()} player action distribution:")
        action_counts = result_df[f"{side}_pred_action_id"].value_counts().sort_index()
        for action_id, count in action_counts.items():
            action_name = id_to_action.get(action_id, f"unknown_{action_id}")
            print(f"    {action_name}: {count} ({count/len(result_df):.2%})")
    
    # Save predictions
    output_path = config.output_dir / f"3d_predictions_fold_{fold}.csv"
    save_3d_predictions(result_df, output_path, config)
    
    return result_df


def run_3d_cross_validation(config: Config3D):
    """Run full cross-validation for 3D pose model"""
    all_metrics = []
    
    for fold in range(config.cross_validation["n_folds"]):
        print(f"\n{'='*60}")
        print(f"3D POSE MODEL - FOLD {fold + 1}/{config.cross_validation['n_folds']}")
        print(f"{'='*60}")
        
        # Train model
        trainer, lit_model, dm = train_3d_model(config, fold)
        
        if trainer is None:
            print(f"Failed to train fold {fold}")
            continue
        
        # Evaluate
        metrics = evaluate_3d_model(trainer, lit_model, dm, config)
        metrics["fold"] = fold
        all_metrics.append(metrics)
        
        # Generate predictions
        generate_3d_predictions(trainer, lit_model, dm, config, fold)
    
    if not all_metrics:
        print("No successful training runs.")
        return
    
    # Save CV results
    cv_results_df = pd.DataFrame(all_metrics)
    cv_results_df.to_csv(config.output_dir / "3d_cv_results.csv", index=False)
    
    # Print CV summary
    print("\n" + "="*60)
    print("3D POSE MODEL CROSS-VALIDATION SUMMARY")
    print("="*60)
    
    for metric in ["overall_accuracy", "overall_accuracy_ignore_none", "overall_f1_macro"]:
        mean_val = cv_results_df[metric].mean()
        std_val = cv_results_df[metric].std()
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Save as JSON
    cv_summary = {
        "mean": cv_results_df.mean().to_dict(),
        "std": cv_results_df.std().to_dict(),
        "per_fold": all_metrics
    }
    
    with open(config.output_dir / "3d_cv_results.json", "w") as f:
        json.dump(cv_summary, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Train 3D pose-based action recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exp08_3d_pose_lstm.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to run (0-based). If not specified, runs all folds"
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Run 3D pose data preparation first"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: reduced data and faster training"
    )
    
    args = parser.parse_args()
    
    # Prepare 3D data if requested
    if args.prepare_data:
        print("Preparing 3D pose data...")
        try:
            from ..preprocess.prepare_3d_data_for_training import main as prepare_main
            prepare_main()
        except ImportError:
            print("Please run the 3D pose preparation script first:")
            print("python src/preprocess/prepare_3d_data_for_training.py")
            return
    
    # Load config
    config = Config3D(args.config)
    config.debug = args.debug
    
    # Debug mode adjustments
    if args.debug:
        config.training["epochs"] = min(config.training["epochs"], 5)
        config.cross_validation["n_folds"] = min(config.cross_validation["n_folds"], 2)
        print("Debug mode enabled: reducing epochs and folds")
    
    # Save config
    config.save(config.output_dir / "config.json")
    
    print(f"3D Pose Model Configuration:")
    print(f"  Model type: {config.model['type']}")
    print(f"  Hidden size: {config.model['hidden_size']}")
    print(f"  Learning rate: {config.training['learning_rate']}")
    print(f"  Batch size: {config.training['batch_size']}")
    print(f"  Epochs: {config.training['epochs']}")
    print(f"  Spatial attention: {config.model['use_spatial_attention']}")
    print(f"  Joint embeddings: {config.model['use_joint_embeddings']}")
    
    # Check if 3D data exists
    if not config.data_dir.exists():
        print(f"\n3D pose data directory not found: {config.data_dir}")
        print("Please run 3D pose data preparation first:")
        print("python src/preprocess/process_pose_preds_3d.py")
        print("python src/preprocess/prepare_3d_data_for_training.py")
        return
    
    # Run training
    if args.fold is not None:
        # Run single fold
        trainer, lit_model, dm = train_3d_model(config, args.fold)
        if trainer is not None:
            metrics = evaluate_3d_model(trainer, lit_model, dm, config)
            generate_3d_predictions(trainer, lit_model, dm, config, args.fold)
    else:
        # Run full cross-validation
        run_3d_cross_validation(config)


if __name__ == "__main__":
    main()