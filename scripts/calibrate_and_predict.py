#!/usr/bin/env python3
"""Example script to calibrate trained models and make predictions with improved probabilities"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.exp07.calibrate_model import calibrate_all_folds
from src.exp07.train import generate_ensemble_predictions
from src.exp07.utils import Config


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
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    print(f"Working with models in: {config.output_dir}")
    
    # Step 1: Calibrate models (if not skipped)
    if not args.skip_calibration:
        print("\n" + "="*60)
        print("STEP 1: CALIBRATING MODELS")
        print("="*60)
        calibrate_all_folds(config)
    else:
        print("\nSkipping calibration, using existing calibration files")
    
    # Step 2: Generate predictions with calibration
    print("\n" + "="*60)
    print("STEP 2: GENERATING PREDICTIONS WITH CALIBRATION")
    print("="*60)
    
    # Generate calibrated ensemble predictions
    calibrated_df = generate_ensemble_predictions(config, use_calibration=True)
    
    # Also generate uncalibrated predictions for comparison
    print("\n" + "="*60)
    # print("GENERATING UNCALIBRATED PREDICTIONS FOR COMPARISON")
    # print("="*60)
    
    # uncalibrated_df = generate_ensemble_predictions(config, use_calibration=False)
    
    # Save both versions
    calibrated_path = config.output_dir / "predictions_ensemble_calibrated.csv"
    calibrated_df.to_csv(calibrated_path, index=False)
    # uncalibrated_path = config.output_dir / "predictions_ensemble_uncalibrated.csv"
    
    print(f"\nCalibrated predictions saved to: {calibrated_path}")
    # print(f"Uncalibrated predictions saved to: {uncalibrated_path}")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("The calibrated predictions should have better-calibrated probabilities,")
    print("especially for the 'none' class which dominates the dataset.")


if __name__ == "__main__":
    main()