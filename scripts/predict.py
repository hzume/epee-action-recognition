#!/usr/bin/env python3
"""Command-line interface for single video prediction."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import load_predictor


def main():
    parser = argparse.ArgumentParser(
        description="Predict fencing actions from video using trained models"
    )
    
    # Required arguments
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )
    
    # Model selection
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="exp04",
        choices=["exp00", "exp01", "exp02", "exp03", "exp04", "exp05"],
        help="Experiment/model to use for prediction (default: exp04)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output CSV file path (default: stdout)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Custom config file path (default: use experiment default)"
    )
    
    parser.add_argument(
        "--checkpoint", "-m",
        type=str,
        default=None,
        help="Custom model checkpoint path (default: use config default)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: auto-detect)"
    )
    
    parser.add_argument(
        "--video-output", "-vo",
        type=str,
        default=None,
        help="Output path for labeled video (e.g., output_labeled.mp4)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)
        
    # Load predictor
    try:
        print(f"Loading {args.experiment} model...", file=sys.stderr)
        predictor = load_predictor(
            args.experiment,
            config_path=args.config,
            checkpoint_path=args.checkpoint
        )
        
        # Override device if specified
        if args.device:
            predictor.device = args.device
            
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Perform prediction
    try:
        print(f"Processing video: {video_path.name}...", file=sys.stderr)
        results = predictor.predict_video(
            video_path, 
            output_path=args.output,
            output_video_path=args.video_output
        )
        
        # Print results if no output file specified
        if args.output is None and args.video_output is None:
            print(results.to_csv(index=False))
        else:
            if args.output:
                print(f"✓ Predictions saved to: {args.output}", file=sys.stderr)
            if args.video_output:
                print(f"✓ Labeled video saved to: {args.video_output}", file=sys.stderr)
            
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()