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
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Temperature for probabilistic sampling (1.0=standard sampling, >1.0=more uniform, <1.0=more peaked)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    
    print(f"Working with models in: {config.output_dir}")
    print(f"Sampling settings: temperature={args.sampling_temperature}, seed={args.random_seed}")
    
    # Step 1: Calibrate models (if not skipped)
    if not args.skip_calibration:
        print("\n" + "="*60)
        print(f"STEP 1: CALIBRATING MODELS WITH {args.method.upper()} SCALING (OPTIMIZING {args.objective.upper()})")
        print("="*60)
        calibrate_all_folds(config, method=args.method, objective=args.objective)
    else:
        print(f"\nSkipping calibration, using existing {args.method} scaling files")
    
    # Step 2: Generate predictions with new sampling approach
    print("\n" + "="*60)
    print(f"STEP 2: GENERATING PREDICTIONS WITH PROBABILISTIC SAMPLING")
    print("="*60)
    
    # Generate predictions using new ensemble → calibration → sampling approach
    calibrated_df = generate_ensemble_predictions(
        config, 
        use_calibration=True, 
        calibration_method=args.method, 
        calibration_objective=args.objective,
        sampling_temperature=args.sampling_temperature,
        random_seed=args.random_seed
    )
    
    # Save predictions with new naming scheme (already handled in generate_ensemble_predictions)
    cal_type = f"{args.method.capitalize()}-{args.objective.upper()}"
    temp_info = f" (temp={args.sampling_temperature})" if args.sampling_temperature != 1.0 else ""
    seed_info = f" (seed={args.random_seed})" if args.random_seed is not None else ""
    
    print(f"\n{cal_type} calibrated predictions with probabilistic sampling{temp_info}{seed_info}")
    print("Output saved by generate_ensemble_predictions()")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("New approach: Ensemble averaging → Calibration → Probabilistic sampling")
    print("This provides theoretically sounder uncertainty quantification.")


if __name__ == "__main__":
    main()