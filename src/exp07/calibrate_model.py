"""Script to calibrate trained exp07 models using temperature scaling"""

import argparse
import warnings
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import EpeeDataset
from .model import ImprovedLSTMModel, LitModel
from .utils import Config, load_normalization_stats
from .calibration import (
    learn_temperature_scaling,
    save_temperature_scaling,
    learn_vector_scaling,
    save_vector_scaling,
    CalibratedModel,
    expected_calibration_error
)

warnings.filterwarnings("ignore")


def calibrate_single_fold(config: Config, fold: int, method: str = "temperature"):
    """Calibrate a single fold model.
    
    Args:
        config: Configuration object
        fold: Fold index to calibrate
        method: Calibration method ('temperature' or 'vector')
    """
    print(f"\n{'='*60}")
    print(f"Calibrating fold {fold} with {method} scaling")
    print(f"{'='*60}")
    
    # Find checkpoint
    fold_dir = config.output_dir / f"fold_{fold}"
    ckpt_files = list(fold_dir.glob("*.ckpt"))
    
    if not ckpt_files:
        print(f"No checkpoint found for fold {fold}")
        return
    
    # Find best checkpoint
    best_ckpt = None
    best_metric = -1
    for ckpt_file in ckpt_files:
        if "last.ckpt" in ckpt_file.name:
            continue
        try:
            parts = ckpt_file.stem.split('-')
            if len(parts) >= 2:
                metric_val = float(parts[1])
                if metric_val > best_metric:
                    best_metric = metric_val
                    best_ckpt = ckpt_file
        except (ValueError, IndexError):
            continue
    
    if best_ckpt is None and ckpt_files:
        best_ckpt = ckpt_files[0]
    
    print(f"Using checkpoint: {best_ckpt}")
    
    # Load normalization stats
    norm_stats_path = fold_dir / "normalization_stats.json"
    if not norm_stats_path.exists():
        print(f"No normalization stats found for fold {fold}")
        return
    
    norm_stats = load_normalization_stats(norm_stats_path)
    
    # Load data
    df = pd.read_csv(config.data_dir / "pose_preds.csv")
    train_val_df = df[~df["video_filename"].isin(config.predict_videos)]
    
    # Get validation split for this fold
    skf = StratifiedGroupKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.seed
    )
    
    stratify_labels = train_val_df["left_action"].fillna("none")
    groups = train_val_df["video_filename"]
    splits = list(skf.split(train_val_df, stratify_labels, groups))
    _, valid_idx = splits[fold]
    valid_df = train_val_df.iloc[valid_idx]
    
    print(f"Validation samples: {len(valid_df)}")
    print(f"Validation videos: {sorted(valid_df['video_filename'].unique())}")
    
    # Create validation dataset WITHOUT balanced sampling (important for calibration!)
    valid_dataset = EpeeDataset(
        valid_df,
        config,
        sample_balanced=False,  # Critical: use actual data distribution
        augment=False,
        normalization_stats=norm_stats
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    num_features = len(valid_dataset.feature_cols)
    model = ImprovedLSTMModel(num_features=num_features, config=config)
    lit_model = LitModel.load_from_checkpoint(
        best_ckpt,
        model=model,
        config=config
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()
    
    # Learn calibration based on method
    if method == "temperature":
        # Learn temperature scaling
        calibration_module = learn_temperature_scaling(
            model=lit_model,
            dataloader=valid_loader,
            device=device,
            max_iter=50,
            lr=0.01
        )
        
        # Save temperature scaling parameters
        temp_path = fold_dir / "temperature_scaling.json"
        save_temperature_scaling(calibration_module, temp_path)
        
        # Also save the state dict for loading in PyTorch (move to CPU first)
        torch.save(calibration_module.cpu().state_dict(), fold_dir / "temperature_scaling.pth")
        
    elif method == "vector":
        # Learn vector scaling
        calibration_module = learn_vector_scaling(
            model=lit_model,
            dataloader=valid_loader,
            device=device,
            num_classes=config.num_classes,
            max_iter=50,
            lr=0.01
        )
        
        # Save vector scaling parameters
        vec_path = fold_dir / "vector_scaling.json"
        save_vector_scaling(calibration_module, vec_path)
        
        # Also save the state dict for loading in PyTorch (move to CPU first)
        torch.save(calibration_module.cpu().state_dict(), fold_dir / "vector_scaling.pth")
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return calibration_module


def calibrate_all_folds(config: Config, method: str = "temperature"):
    """Calibrate all fold models."""
    for fold in range(config.n_folds):
        try:
            calibrate_single_fold(config, fold, method)
        except Exception as e:
            print(f"Failed to calibrate fold {fold}: {e}")


def evaluate_calibration_improvement(config: Config):
    """Evaluate the improvement from calibration on all folds."""
    print("\n" + "="*60)
    print("CALIBRATION IMPROVEMENT SUMMARY")
    print("="*60)
    
    improvements = []
    
    for fold in range(config.n_folds):
        fold_dir = config.output_dir / f"fold_{fold}"
        temp_path = fold_dir / "temperature_scaling.json"
        
        if temp_path.exists():
            # Temperature values indicate improvement
            import json
            with open(temp_path, 'r') as f:
                temps = json.load(f)
            
            print(f"\nFold {fold}:")
            print(f"  Temperature Left:  {temps['temperature_left']:.4f}")
            print(f"  Temperature Right: {temps['temperature_right']:.4f}")
            
            # Values significantly different from 1.0 indicate calibration was needed
            avg_temp_diff = (abs(temps['temperature_left'] - 1.0) + abs(temps['temperature_right'] - 1.0)) / 2
            improvements.append(avg_temp_diff)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nAverage temperature deviation from 1.0: {avg_improvement:.4f}")
        print("(Larger deviations indicate more calibration was needed)")


def main():
    parser = argparse.ArgumentParser(description="Calibrate trained exp07 models")
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
        help="Specific fold to calibrate (0-based). If not specified, calibrates all folds"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory containing trained models"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="temperature",
        choices=["temperature", "vector"],
        help="Calibration method to use (default: temperature)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    print(f"Configuration:")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Number of folds: {config.n_folds}")
    
    # Check if output directory exists
    if not config.output_dir.exists():
        raise ValueError(f"Output directory not found: {config.output_dir}")
    
    # Calibrate models
    if args.fold is not None:
        # Calibrate single fold
        calibrate_single_fold(config, args.fold, args.method)
    else:
        # Calibrate all folds
        calibrate_all_folds(config, args.method)
        
        # Show improvement summary
        evaluate_calibration_improvement(config)
    
    print("\nCalibration complete!")
    print("The calibrated models can now be used for prediction with improved probability estimates.")


if __name__ == "__main__":
    main()