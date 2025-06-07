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
    learn_temperature_scaling_f1,
    save_temperature_scaling,
    learn_vector_scaling,
    learn_vector_scaling_f1,
    save_vector_scaling,
    learn_distribution_calibration,
    save_distribution_calibration,
    CalibratedModel,
    DistributionCalibratedModel,
    expected_calibration_error
)

warnings.filterwarnings("ignore")


def calibrate_single_fold(config: Config, fold: int, method: str = "temperature", objective: str = "ece", use_distribution_calibration: bool = False, distribution_only: bool = False):
    """Calibrate a single fold model.
    
    Args:
        config: Configuration object
        fold: Fold index to calibrate
        method: Calibration method ('temperature' or 'vector')
        objective: Optimization objective ('ece' or 'f1')
        use_distribution_calibration: Whether to apply distribution calibration
        distribution_only: If True, only run distribution calibration using existing probability calibration
    """
    if distribution_only:
        print(f"\n{'='*60}")
        print(f"Running DISTRIBUTION CALIBRATION ONLY for fold {fold} (using existing {method} {objective} calibration)")
        print(f"{'='*60}")
    else:
        dist_suffix = " + Distribution" if use_distribution_calibration else ""
        print(f"\n{'='*60}")
        print(f"Calibrating fold {fold} with {method} scaling{dist_suffix} (optimizing {objective.upper()})")
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
    
    # Handle distribution-only mode
    calibration_module = None
    if distribution_only:
        # Skip probability calibration entirely - use raw model outputs
        print(f"Distribution-only mode: Skipping probability calibration, using raw model outputs")
        calibration_module = None
    else:
        # Learn calibration based on method and objective
        if method == "temperature":
            if objective == "ece":
                # Learn temperature scaling optimizing ECE
                calibration_module = learn_temperature_scaling(
                    model=lit_model,
                    dataloader=valid_loader,
                    device=device,
                    max_iter=50,
                    lr=0.01
                )
            elif objective == "f1":
                # Learn temperature scaling optimizing F1 score
                calibration_module = learn_temperature_scaling_f1(
                    model=lit_model,
                    dataloader=valid_loader,
                    device=device,
                    max_iter=50,
                    lr=0.01
                )
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            # Save temperature scaling parameters with objective suffix
            suffix = f"_{objective}" if objective != "ece" else ""
            temp_path = fold_dir / f"temperature_scaling{suffix}.json"
            save_temperature_scaling(calibration_module, temp_path)
            
            # Also save the state dict for loading in PyTorch (move to CPU first)
            torch.save(calibration_module.cpu().state_dict(), fold_dir / f"temperature_scaling{suffix}.pth")
            
        elif method == "vector":
            if objective == "ece":
                # Learn vector scaling optimizing ECE
                calibration_module = learn_vector_scaling(
                    model=lit_model,
                    dataloader=valid_loader,
                    device=device,
                    num_classes=config.num_classes,
                    max_iter=50,
                    lr=0.01
                )
            elif objective == "f1":
                # Learn vector scaling optimizing F1 score
                calibration_module = learn_vector_scaling_f1(
                    model=lit_model,
                    dataloader=valid_loader,
                    device=device,
                    num_classes=config.num_classes,
                    max_iter=50,
                    lr=0.01
                )
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            # Save vector scaling parameters with objective suffix
            suffix = f"_{objective}" if objective != "ece" else ""
            vec_path = fold_dir / f"vector_scaling{suffix}.json"
            save_vector_scaling(calibration_module, vec_path)
            
            # Also save the state dict for loading in PyTorch (move to CPU first)
            torch.save(calibration_module.cpu().state_dict(), fold_dir / f"vector_scaling{suffix}.pth")
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    # Apply distribution calibration if requested (or if distribution_only mode)
    dist_calibration = None
    if use_distribution_calibration or distribution_only:
        print(f"\n{'='*40}")
        print("LEARNING DISTRIBUTION CALIBRATION")
        print(f"{'='*40}")
        
        # Learn distribution calibration
        prob_calibration_module = calibration_module.to(device) if calibration_module is not None else None
        dist_calibration = learn_distribution_calibration(
            model=lit_model,
            dataloader=valid_loader,
            device=device,
            num_classes=config.num_classes,
            prob_calibration_module=prob_calibration_module
        )
        
        # Save distribution calibration parameters
        dist_suffix = f"_{objective}" if objective != "ece" else ""
        if distribution_only:
            # For distribution-only mode, save with special naming
            dist_path = fold_dir / f"distribution_calibration_only{dist_suffix}.json"
        else:
            dist_path = fold_dir / f"distribution_calibration_{method}{dist_suffix}.json"
        save_distribution_calibration(dist_calibration, dist_path)
    
    return calibration_module, dist_calibration


def calibrate_all_folds(config: Config, method: str = "temperature", objective: str = "ece", use_distribution_calibration: bool = False, distribution_only: bool = False):
    """Calibrate all fold models."""
    for fold in range(config.n_folds):
        try:
            calibrate_single_fold(config, fold, method, objective, use_distribution_calibration, distribution_only)
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
    parser.add_argument(
        "--objective",
        type=str,
        default="ece",
        choices=["ece", "f1"],
        help="Optimization objective: 'ece' for calibration or 'f1' for classification performance (default: ece)"
    )
    parser.add_argument(
        "--distribution_calibration",
        action="store_true",
        help="Apply distribution calibration (threshold adjustment) after probability calibration"
    )
    parser.add_argument(
        "--distribution_only",
        action="store_true",
        help="Run only distribution calibration using existing probability calibration (must specify --method and --objective to match existing calibration)"
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
    
    # Validate distribution_only option
    if args.distribution_only:
        if not args.distribution_calibration:
            # Force distribution_calibration to True when using distribution_only
            args.distribution_calibration = True
        print(f"Running DISTRIBUTION CALIBRATION ONLY mode:")
        print(f"  Will load existing {args.method} {args.objective} calibration and apply distribution calibration")
    
    # Calibrate models
    if args.fold is not None:
        # Calibrate single fold
        calibrate_single_fold(config, args.fold, args.method, args.objective, args.distribution_calibration, args.distribution_only)
    else:
        # Calibrate all folds
        calibrate_all_folds(config, args.method, args.objective, args.distribution_calibration, args.distribution_only)
        
        # Show improvement summary (skip for distribution_only mode)
        if not args.distribution_only:
            evaluate_calibration_improvement(config)
    
    print("\nCalibration complete!")
    print("The calibrated models can now be used for prediction with improved probability estimates.")


if __name__ == "__main__":
    main()