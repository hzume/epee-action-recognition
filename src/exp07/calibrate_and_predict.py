#!/usr/bin/env python3
"""Calibrate trained models and make predictions with improved probabilities"""

import argparse
from pathlib import Path

from .calibrate_model import calibrate_all_folds
from .train import generate_ensemble_predictions
from .utils import Config


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate exp07 models and generate predictions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory containing trained models"
    )
    parser.add_argument(
        "--skip_calibration",
        action="store_true",
        help="Skip calibration and use existing calibration files"
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
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    print(f"Working with models in: {config.output_dir}")
    
    # Step 1: Calibrate models (if not skipped)
    if not args.skip_calibration:
        dist_suffix = " + DISTRIBUTION" if args.distribution_calibration else ""
        print("\n" + "="*60)
        print(f"STEP 1: CALIBRATING MODELS WITH {args.method.upper()} SCALING{dist_suffix} (OPTIMIZING {args.objective.upper()})")
        print("="*60)
        calibrate_all_folds(config, method=args.method, objective=args.objective, use_distribution_calibration=args.distribution_calibration)
    else:
        print(f"\nSkipping calibration, using existing {args.method} scaling files")
    
    # Step 2: Generate predictions with calibration
    print("\n" + "="*60)
    print(f"STEP 2: GENERATING PREDICTIONS WITH {args.method.upper()} CALIBRATION")
    print("="*60)
    
    # Generate calibrated ensemble predictions
    calibrated_df = generate_ensemble_predictions(config, use_calibration=True, calibration_method=args.method, calibration_objective=args.objective, use_distribution_calibration=args.distribution_calibration)
    
    # Also generate uncalibrated predictions for comparison
    print("\n" + "="*60)
    # print("GENERATING UNCALIBRATED PREDICTIONS FOR COMPARISON")
    # print("="*60)
    
    # uncalibrated_df = generate_ensemble_predictions(config, use_calibration=False)
    
    # Save predictions with method, objective, and distribution calibration in filename
    dist_suffix = "_dist" if args.distribution_calibration else ""
    calibrated_path = config.output_dir / f"predictions_ensemble_{args.method}_{args.objective}{dist_suffix}_calibrated.csv"
    calibrated_df.to_csv(calibrated_path, index=False)
    
    cal_type = f"{args.method.capitalize()}-{args.objective.upper()}"
    if args.distribution_calibration:
        cal_type += "+Distribution"
    print(f"\n{cal_type} calibrated predictions saved to: {calibrated_path}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("The calibrated predictions should have better-calibrated probabilities,")
    print("especially for the 'none' class which dominates the dataset.")


if __name__ == "__main__":
    main()