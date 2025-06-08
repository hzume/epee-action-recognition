"""Process 3D pose predictions from MMPose output"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


def calculate_3d_bbox_from_keypoints(keypoints_3d):
    """Calculate a simple 3D bounding box from keypoints"""
    keypoints_array = np.array(keypoints_3d)
    
    # Calculate min/max for each axis
    x_min, x_max = keypoints_array[:, 0].min(), keypoints_array[:, 0].max()
    y_min, y_max = keypoints_array[:, 1].min(), keypoints_array[:, 1].max()
    z_min, z_max = keypoints_array[:, 2].min(), keypoints_array[:, 2].max()
    
    # Return bounding box in format [x_min, y_min, x_max, y_max] for 2D compatibility
    # and add 3D extent as a simple measure
    bbox_2d = [x_min, y_min, x_max, y_max]
    bbox_3d_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    return bbox_2d, bbox_3d_volume

# Directories
data_dir = Path("input/data_10hz")
preds_3d_dir = Path("output/pose_10hz_3d")

# Load frame labels
frame_label_df = pd.read_csv(data_dir / "frame_label.csv")

# Initialize data dictionary for 3D poses
pose_3d_data = {
    "frame_filename": [],
    "video_filename": [],
    "frame_idx": [],
    "labels": [],
    "left_action": [],
    "left_outcome": [],
    "right_action": [],
    "right_outcome": [],
    "instance_id": [],
    "width": [],
    "height": [],
} | {f"keypoint_{i}_x": [] for i in range(17)} \
    | {f"keypoint_{i}_y": [] for i in range(17)} \
    | {f"keypoint_{i}_z": [] for i in range(17)} \
    | {f"keypoint_score_{i}": [] for i in range(17)} \
    | {"bbox_x_1": [], "bbox_y_1": [], "bbox_x_2": [], "bbox_y_2": [], "bbox_3d_volume": []}

# Process each frame
for _, row in tqdm(frame_label_df.iterrows(), total=len(frame_label_df)):
    frame_filename = row["frame_filename"]
    frame_basename = Path(frame_filename).stem
    
    # Load 3D predictions
    pred_3d_path = preds_3d_dir / f"{frame_basename}.json"
    if not pred_3d_path.exists():
        print(f"Warning: 3D pose file not found: {pred_3d_path}")
        continue
        
    with open(pred_3d_path, "r") as f:
        preds_3d = json.load(f)
    
    # Process each detected instance
    for i, instance_3d in enumerate(preds_3d):
        # Extract 3D keypoints and scores
        keypoints_3d = instance_3d["keypoints"]
        keypoint_scores = instance_3d["keypoint_scores"]
        
        # Calculate bounding box from 3D keypoints
        bbox_2d, bbox_3d_volume = calculate_3d_bbox_from_keypoints(keypoints_3d)
        
        # Add frame metadata
        pose_3d_data["frame_filename"].append(frame_filename)
        pose_3d_data["video_filename"].append(row["video_filename"])
        pose_3d_data["frame_idx"].append(row["frame_idx"])
        pose_3d_data["labels"].append(row["labels"])
        
        # Process action labels
        left_actions = row["left_actions"]
        right_actions = row["right_actions"]
        left_outcomes = row["left_outcomes"]
        right_outcomes = row["right_outcomes"]
        
        # Handle action labels (extract first action if multiple)
        pose_3d_data["left_action"].append(
            left_actions.split(",")[0] if isinstance(left_actions, str) else None
        )
        pose_3d_data["right_action"].append(
            right_actions.split(",")[0] if isinstance(right_actions, str) else None
        )
        
        # Handle outcome labels
        pose_3d_data["left_outcome"].append(
            left_outcomes.split(",")[0] if isinstance(left_outcomes, str) else None
        )
        pose_3d_data["right_outcome"].append(
            right_outcomes.split(",")[0] if isinstance(right_outcomes, str) else None
        )
        
        # Add instance information
        pose_3d_data["instance_id"].append(i)
        pose_3d_data["width"].append(row["width"])
        pose_3d_data["height"].append(row["height"])
        
        # Add 3D keypoints
        for j, (keypoint_3d, score) in enumerate(zip(keypoints_3d, keypoint_scores)):
            pose_3d_data[f"keypoint_{j}_x"].append(keypoint_3d[0])
            pose_3d_data[f"keypoint_{j}_y"].append(keypoint_3d[1])
            pose_3d_data[f"keypoint_{j}_z"].append(keypoint_3d[2])
            pose_3d_data[f"keypoint_score_{j}"].append(score)
        
        # Add calculated bounding box information
        pose_3d_data["bbox_x_1"].append(bbox_2d[0])
        pose_3d_data["bbox_y_1"].append(bbox_2d[1])
        pose_3d_data["bbox_x_2"].append(bbox_2d[2])
        pose_3d_data["bbox_y_2"].append(bbox_2d[3])
        pose_3d_data["bbox_3d_volume"].append(bbox_3d_volume)

# Create DataFrame
pose_3d_df = pd.DataFrame(pose_3d_data)

# Display summary statistics
print("\n=== 3D Pose Data Summary ===")
print(f"Total frames processed: {len(pose_3d_df['frame_filename'].unique())}")
print(f"Total instances: {len(pose_3d_df)}")
print(f"Videos: {sorted(pose_3d_df['video_filename'].unique())}")

# Check data quality
print("\n=== Data Quality Check ===")
print(f"Frames with multiple instances: {(pose_3d_df.groupby('frame_filename').size() > 1).sum()}")
print(f"Average instances per frame: {pose_3d_df.groupby('frame_filename').size().mean():.2f}")

# Check 3D coordinate ranges
print("\n=== 3D Coordinate Ranges ===")
for axis in ['x', 'y', 'z']:
    coords = []
    for i in range(17):
        coords.extend(pose_3d_df[f"keypoint_{i}_{axis}"].values)
    coords = np.array(coords)
    print(f"{axis.upper()}-axis: min={coords.min():.3f}, max={coords.max():.3f}, mean={coords.mean():.3f}")

# Check score statistics
all_scores = []
for i in range(17):
    all_scores.extend(pose_3d_df[f"keypoint_score_{i}"].values)
all_scores = np.array(all_scores)
print(f"\nKeypoint scores: min={all_scores.min():.3f}, max={all_scores.max():.3f}, mean={all_scores.mean():.3f}")

# Check 3D bounding box statistics
print(f"\n=== 3D Bounding Box Statistics ===")
bbox_volumes = pose_3d_df["bbox_3d_volume"].values
print(f"3D Bbox volume: min={bbox_volumes.min():.3f}, max={bbox_volumes.max():.3f}, mean={bbox_volumes.mean():.3f}")

# Calculate bbox areas (2D projection)
bbox_areas = (pose_3d_df["bbox_x_2"] - pose_3d_df["bbox_x_1"]) * (pose_3d_df["bbox_y_2"] - pose_3d_df["bbox_y_1"])
print(f"2D Bbox area: min={bbox_areas.min():.3f}, max={bbox_areas.max():.3f}, mean={bbox_areas.mean():.3f}")

# Display sample data
print("\n=== Sample Data (first 5 rows) ===")
print(pose_3d_df.head())

# Save to CSV
output_path = data_dir / "pose_preds_3d.csv"
pose_3d_df.to_csv(output_path, index=False)
print(f"\n3D pose predictions saved to: {output_path}")

print("\n=== Processing Complete ===")