"""Video writing utilities for visualizing predictions."""

from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np
import pandas as pd


def create_labeled_video(
    video_path: Union[str, Path],
    predictions_df: pd.DataFrame,
    output_path: Union[str, Path],
    fps: int = 10,
    font_scale: float = 1.0,
    font_thickness: int = 2,
    left_color: tuple = (0, 0, 255),  # Red in BGR
    right_color: tuple = (0, 255, 0),  # Green in BGR
) -> None:
    """Create a video with prediction labels overlaid.
    
    Args:
        video_path: Path to input video
        predictions_df: DataFrame with predictions (must have columns: frame_idx, left_pred_action, right_pred_action)
        output_path: Path for output video
        fps: Prediction frame rate (default: 10 Hz)
        font_scale: Font size scale
        font_thickness: Font thickness
        left_color: Color for left player labels (BGR)
        right_color: Color for right player labels (BGR)
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate frame interval for predictions
    frame_interval = int(video_fps / fps)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))
    
    # Prepare prediction data
    frame_indices = predictions_df["frame_idx"].values
    left_actions = predictions_df["left_pred_action"].values
    right_actions = predictions_df["right_pred_action"].values
    
    # Process video
    frame_count = 0
    pred_frame_count = 0
    current_left_action = ""
    current_right_action = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update labels at prediction intervals
        if frame_count % frame_interval == 0:
            if pred_frame_count in frame_indices:
                idx = np.where(frame_indices == pred_frame_count)[0][0]
                current_left_action = str(left_actions[idx])
                current_right_action = str(right_actions[idx])
            else:
                current_left_action = ""
                current_right_action = ""
            pred_frame_count += 1
        
        # Draw labels on frame
        cv2.putText(
            img=frame,
            text=f"left: {current_left_action}",
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=left_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA
        )
        
        cv2.putText(
            img=frame,
            text=f"right: {current_right_action}",
            org=(50, 100),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=right_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA
        )
        
        # Write frame
        writer.write(frame)
        frame_count += 1
    
    # Clean up
    cap.release()
    writer.release()
    
    print(f"✓ Output video saved: {output_path}")


def create_labeled_video_batch(
    predictions_csv: Union[str, Path],
    output_dir: Union[str, Path],
    video_dir: Optional[Union[str, Path]] = None,
    fps: int = 10,
) -> None:
    """Create labeled videos for all predictions in a CSV file.
    
    Args:
        predictions_csv: Path to CSV with predictions
        output_dir: Directory for output videos
        video_dir: Directory containing input videos (default: input/videos)
        fps: Prediction frame rate
    """
    predictions_csv = Path(predictions_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if video_dir is None:
        video_dir = Path("input/videos")
    else:
        video_dir = Path(video_dir)
    
    # Load predictions
    df = pd.read_csv(predictions_csv)
    
    # Process each video
    video_filenames = df["video_filename"].unique()
    for video_filename in video_filenames:
        print(f"Processing: {video_filename}")
        
        # Get predictions for this video
        video_df = df[df["video_filename"] == video_filename]
        
        # Find video path
        video_path = video_dir / video_filename
        if not video_path.exists():
            print(f"  ✗ Video not found: {video_path}")
            continue
        
        # Create output path
        output_path = output_dir / video_filename
        
        try:
            create_labeled_video(
                video_path=video_path,
                predictions_df=video_df,
                output_path=output_path,
                fps=fps
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")