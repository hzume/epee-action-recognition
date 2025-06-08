#!/usr/bin/env python3
"""Command-line interface for batch video prediction with video output."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference import load_predictor
from src.inference.utils import create_labeled_video_batch


def main():
    parser = argparse.ArgumentParser(
        description="Predict fencing actions from multiple videos with optional video output"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video-dir", "-d",
        type=str,
        help="Directory containing video files"
    )
    input_group.add_argument(
        "--video-list", "-l",
        type=str,
        nargs="+",
        help="List of video file paths"
    )
    
    # Model selection
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="exp04",
        choices=["exp00", "exp01", "exp02", "exp03", "exp04", "exp05"],
        help="Experiment/model to use for prediction (default: exp04)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for prediction CSV files"
    )
    
    parser.add_argument(
        "--video-output", "-vo",
        action="store_true",
        help="Generate labeled output videos"
    )
    
    parser.add_argument(
        "--video-output-dir", "-vod",
        type=str,
        default=None,
        help="Directory for labeled videos (default: output-dir/videos)"
    )
    
    # Optional arguments
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
        "--pattern",
        type=str,
        default="*.mp4",
        help="File pattern for video directory (default: *.mp4)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have predictions"
    )
    
    args = parser.parse_args()
    
    # Collect video paths
    video_paths = []
    if args.video_dir:
        video_dir = Path(args.video_dir)
        if not video_dir.exists():
            print(f"Error: Directory not found: {video_dir}", file=sys.stderr)
            sys.exit(1)
        video_paths = list(video_dir.glob(args.pattern))
    else:
        video_paths = [Path(p) for p in args.video_list]
        
    # Validate video paths
    valid_paths = []
    for path in video_paths:
        if not path.exists():
            print(f"Warning: Video not found, skipping: {path}", file=sys.stderr)
        else:
            valid_paths.append(path)
            
    if not valid_paths:
        print("Error: No valid video files found", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found {len(valid_paths)} video(s) to process", file=sys.stderr)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter existing if requested
    if args.skip_existing:
        filtered_paths = []
        for path in valid_paths:
            output_path = output_dir / f"{path.stem}_predictions.csv"
            if output_path.exists():
                print(f"Skipping existing: {path.name}", file=sys.stderr)
            else:
                filtered_paths.append(path)
        valid_paths = filtered_paths
        
    if not valid_paths:
        print("All videos already processed", file=sys.stderr)
        sys.exit(0)
        
    # Load predictor
    try:
        print(f"\nLoading {args.experiment} model...", file=sys.stderr)
        predictor = load_predictor(
            args.experiment,
            config_path=args.config,
            checkpoint_path=args.checkpoint
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Process videos
    print(f"\nProcessing {len(valid_paths)} video(s)...\n", file=sys.stderr)
    
    results = predictor.predict_batch_videos(valid_paths, output_dir)
    
    # Summary
    successful = sum(1 for r in results.values() if r is not None)
    failed = len(results) - successful
    
    print(f"\nProcessing complete:", file=sys.stderr)
    print(f"  ✓ Successful: {successful}", file=sys.stderr)
    if failed > 0:
        print(f"  ✗ Failed: {failed}", file=sys.stderr)
        
    print(f"\nResults saved to: {output_dir}", file=sys.stderr)
    
    # Generate videos if requested
    if args.video_output and successful > 0:
        # Combine all results
        import pandas as pd
        combined = []
        for video_name, df in results.items():
            if df is not None:
                combined.append(df)
                
        if combined:
            all_results = pd.concat(combined, ignore_index=True)
            combined_path = output_dir / "all_predictions.csv"
            all_results.to_csv(combined_path, index=False)
            
            # Set video output directory
            video_output_dir = args.video_output_dir
            if video_output_dir is None:
                video_output_dir = output_dir / "videos"
            else:
                video_output_dir = Path(video_output_dir)
                
            print(f"\nGenerating labeled videos...", file=sys.stderr)
            
            # Determine video directory (where source videos are)
            if args.video_dir:
                source_video_dir = Path(args.video_dir)
            else:
                # For video list, find common parent directory
                source_video_dir = None
                
            create_labeled_video_batch(
                predictions_csv=combined_path,
                output_dir=video_output_dir,
                video_dir=source_video_dir,
                fps=predictor.config.get("fps", 10)
            )


if __name__ == "__main__":
    main()