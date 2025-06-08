"""Prepare 3D pose data for training by creating appropriate data splits and features"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def calculate_bbox_volume(row):
    """Calculate 3D bounding box volume from keypoints"""
    # Get all keypoint coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    
    for i in range(17):  # 17 keypoints in Human3D format
        x_coords.append(row[f'keypoint_{i}_x'])
        y_coords.append(row[f'keypoint_{i}_y'])
        z_coords.append(row[f'keypoint_{i}_z'])
    
    # Calculate bounding box dimensions
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    
    # Calculate volume
    width = x_max - x_min
    height = y_max - y_min
    depth = z_max - z_min
    volume = width * height * depth
    
    return volume, width, height, depth


def filter_3d_data(pose_3d_df, depth_min=None, depth_max=None, volume_min=None, volume_max=None):
    """Filter 3D pose data by depth and volume ranges, keeping only frames with exactly 2 instances"""
    print("\n=== Filtering 3D Pose Data ===")
    original_count = len(pose_3d_df)
    print(f"Original data: {original_count} instances")
    
    df_filtered = pose_3d_df.copy()
    
    # Apply depth filtering
    if depth_min is not None or depth_max is not None:
        print(f"Applying depth filtering: min={depth_min}, max={depth_max}")
        if depth_min is not None:
            df_filtered = df_filtered[df_filtered['keypoint_0_z'] >= depth_min]
        if depth_max is not None:
            df_filtered = df_filtered[df_filtered['keypoint_0_z'] <= depth_max]
        
        depth_filtered_count = len(df_filtered)
        print(f"After depth filtering: {depth_filtered_count} instances ({depth_filtered_count/original_count*100:.1f}%)")
    
    # Apply volume filtering
    if volume_min is not None or volume_max is not None:
        print(f"Applying volume filtering: min={volume_min}, max={volume_max}")
        pre_volume_count = len(df_filtered)
        
        # Calculate volumes for remaining instances
        valid_indices = []
        for idx, row in df_filtered.iterrows():
            volume, _, _, _ = calculate_bbox_volume(row)
            
            volume_valid = True
            if volume_min is not None and volume < volume_min:
                volume_valid = False
            if volume_max is not None and volume > volume_max:
                volume_valid = False
            
            if volume_valid:
                valid_indices.append(idx)
        
        # Filter by volume
        df_filtered = df_filtered.loc[valid_indices]
        volume_filtered_count = len(df_filtered)
        print(f"After volume filtering: {volume_filtered_count} instances ({volume_filtered_count/pre_volume_count*100:.1f}%)")
    
    # Filter to keep only frames with exactly 2 instances
    print("Filtering to frames with exactly 2 instances...")
    frame_counts = {}
    for _, row in df_filtered.iterrows():
        video_frame_key = (row['video_filename'], row['frame_idx'])
        frame_counts[video_frame_key] = frame_counts.get(video_frame_key, 0) + 1
    
    # Find frames with exactly 2 instances
    frames_with_2_instances = {key for key, count in frame_counts.items() if count == 2}
    
    # Filter data to only include these frames
    final_indices = []
    for idx, row in df_filtered.iterrows():
        video_frame_key = (row['video_filename'], row['frame_idx'])
        if video_frame_key in frames_with_2_instances:
            final_indices.append(idx)
    
    df_final = df_filtered.loc[final_indices]
    
    print(f"Frames with exactly 2 instances: {len(frames_with_2_instances)}")
    print(f"Final filtered data: {len(df_final)} instances ({len(df_final)/original_count*100:.1f}% of original)")
    
    # Show instance distribution analysis
    final_frame_counts = {}
    for _, row in df_final.iterrows():
        video_frame_key = (row['video_filename'], row['frame_idx'])
        final_frame_counts[video_frame_key] = final_frame_counts.get(video_frame_key, 0) + 1
    
    final_counts = list(final_frame_counts.values())
    if final_counts:
        unique_counts, count_freq = np.unique(final_counts, return_counts=True)
        print(f"\nInstance count verification:")
        for count, freq in zip(unique_counts, count_freq):
            print(f"  {count} instances: {freq} frames ({freq/len(final_counts)*100:.1f}%)")
    
    return df_final


def create_3d_features(pose_3d_df):
    """Create additional 3D-specific features for training"""
    print("Creating 3D-specific features...")
    
    # Copy dataframe
    df = pose_3d_df.copy()
    
    # 1. Add joint angles in 3D space
    def calculate_3d_angle(p1, p2, p3):
        """Calculate angle between three 3D points with p2 as vertex"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Handle zero vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        # Clip to handle numerical errors
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return angle
    
    # Define joints for angle calculation (Human3D format)
    angle_joints = [
        ("left_elbow_angle", 11, 12, 13),   # left shoulder-elbow-wrist
        ("right_elbow_angle", 14, 15, 16),  # right shoulder-elbow-wrist
        ("left_knee_angle", 1, 2, 3),       # left hip-knee-ankle
        ("right_knee_angle", 4, 5, 6),      # right hip-knee-ankle
        ("left_shoulder_angle", 8, 11, 12), # thorax-shoulder-elbow
        ("right_shoulder_angle", 8, 14, 15), # thorax-shoulder-elbow
        ("left_hip_angle", 0, 1, 2),        # pelvis-hip-knee
        ("right_hip_angle", 0, 4, 5),       # pelvis-hip-knee
        ("spine_angle", 0, 7, 8),           # pelvis-spine-thorax
        ("neck_angle", 8, 9, 10),           # thorax-neck-head
    ]
    
    for angle_name, j1, j2, j3 in angle_joints:
        angles = []
        for idx in df.index:
            p1 = np.array([df.loc[idx, f"keypoint_{j1}_x"], 
                          df.loc[idx, f"keypoint_{j1}_y"], 
                          df.loc[idx, f"keypoint_{j1}_z"]])
            p2 = np.array([df.loc[idx, f"keypoint_{j2}_x"], 
                          df.loc[idx, f"keypoint_{j2}_y"], 
                          df.loc[idx, f"keypoint_{j2}_z"]])
            p3 = np.array([df.loc[idx, f"keypoint_{j3}_x"], 
                          df.loc[idx, f"keypoint_{j3}_y"], 
                          df.loc[idx, f"keypoint_{j3}_z"]])
            
            angle = calculate_3d_angle(p1, p2, p3)
            angles.append(angle)
        
        df[angle_name] = angles
    
    # 2. Add velocity features (if sequential data is available)
    print("Adding velocity features...")
    df = df.sort_values(['video_filename', 'frame_idx'])
    
    # Initialize velocity columns
    for i in range(17):
        df[f"keypoint_{i}_vx"] = 0.0
        df[f"keypoint_{i}_vy"] = 0.0
        df[f"keypoint_{i}_vz"] = 0.0
    
    # Calculate velocities within each video
    for video in df['video_filename'].unique():
        video_mask = df['video_filename'] == video
        video_indices = df[video_mask].index
        
        for i in range(1, len(video_indices)):
            curr_idx = video_indices[i]
            prev_idx = video_indices[i-1]
            
            # Check if consecutive frames
            if df.loc[curr_idx, 'frame_idx'] == df.loc[prev_idx, 'frame_idx'] + 1:
                for kp in range(17):
                    df.loc[curr_idx, f"keypoint_{kp}_vx"] = (
                        df.loc[curr_idx, f"keypoint_{kp}_x"] - 
                        df.loc[prev_idx, f"keypoint_{kp}_x"]
                    )
                    df.loc[curr_idx, f"keypoint_{kp}_vy"] = (
                        df.loc[curr_idx, f"keypoint_{kp}_y"] - 
                        df.loc[prev_idx, f"keypoint_{kp}_y"]
                    )
                    df.loc[curr_idx, f"keypoint_{kp}_vz"] = (
                        df.loc[curr_idx, f"keypoint_{kp}_z"] - 
                        df.loc[prev_idx, f"keypoint_{kp}_z"]
                    )
    
    # 4. Add body orientation features (Human3D format)
    print("Adding body orientation features...")
    
    # Calculate torso vector (from pelvis to thorax)
    for idx in df.index:
        # Pelvis position (keypoint 0)
        pelvis = np.array([
            df.loc[idx, "keypoint_0_x"],
            df.loc[idx, "keypoint_0_y"],
            df.loc[idx, "keypoint_0_z"]
        ])
        
        # Thorax position (keypoint 8)
        thorax = np.array([
            df.loc[idx, "keypoint_8_x"],
            df.loc[idx, "keypoint_8_y"],
            df.loc[idx, "keypoint_8_z"]
        ])
        
        # Head position (keypoint 10) for additional orientation
        head = np.array([
            df.loc[idx, "keypoint_10_x"],
            df.loc[idx, "keypoint_10_y"],
            df.loc[idx, "keypoint_10_z"]
        ])
        
        # Torso vector (pelvis to thorax)
        torso_vector = thorax - pelvis
        torso_length = np.linalg.norm(torso_vector)
        
        # Head vector (thorax to head)
        head_vector = head - thorax
        head_length = np.linalg.norm(head_vector)
        
        if torso_length > 0:
            torso_vector = torso_vector / torso_length
            
            # Store torso orientation angles
            df.loc[idx, "torso_pitch"] = np.arcsin(np.clip(torso_vector[1], -1, 1))  # Vertical tilt
            df.loc[idx, "torso_yaw"] = np.arctan2(torso_vector[0], torso_vector[2])  # Horizontal rotation
        else:
            df.loc[idx, "torso_pitch"] = 0.0
            df.loc[idx, "torso_yaw"] = 0.0
        
        if head_length > 0:
            head_vector = head_vector / head_length
            
            # Store head orientation angles
            df.loc[idx, "head_pitch"] = np.arcsin(np.clip(head_vector[1], -1, 1))  # Head vertical tilt
            df.loc[idx, "head_yaw"] = np.arctan2(head_vector[0], head_vector[2])  # Head horizontal rotation
        else:
            df.loc[idx, "head_pitch"] = 0.0
            df.loc[idx, "head_yaw"] = 0.0
        
        # Shoulder span vector (left to right shoulder)
        left_shoulder = np.array([
            df.loc[idx, "keypoint_11_x"],
            df.loc[idx, "keypoint_11_y"],
            df.loc[idx, "keypoint_11_z"]
        ])
        right_shoulder = np.array([
            df.loc[idx, "keypoint_14_x"],
            df.loc[idx, "keypoint_14_y"],
            df.loc[idx, "keypoint_14_z"]
        ])
        
        shoulder_vector = right_shoulder - left_shoulder
        shoulder_span = np.linalg.norm(shoulder_vector)
        df.loc[idx, "shoulder_span"] = shoulder_span
        
        if shoulder_span > 0:
            shoulder_vector = shoulder_vector / shoulder_span
            # Shoulder rotation (roll angle)
            df.loc[idx, "shoulder_roll"] = np.arctan2(shoulder_vector[1], shoulder_vector[0])
        else:
            df.loc[idx, "shoulder_roll"] = 0.0
    
    return df


def prepare_training_data(pose_3d_df, test_videos=None, depth_min=None, depth_max=None, volume_min=None, volume_max=None, output_suffix=""):
    """Prepare data for training with proper splits"""
    
    # Apply filtering first
    df = filter_3d_data(pose_3d_df, depth_min, depth_max, volume_min, volume_max)
    
    if len(df) == 0:
        print("ERROR: No data remains after filtering!")
        return None, None, None, None
    
    # Create enhanced features
    df = create_3d_features(df)
    
    # Define test videos if not provided
    if test_videos is None:
        test_videos = [
            '2024-11-10-18-33-49.mp4',
            '2024-11-10-19-21-45.mp4',
            '2025-01-04_08-37-18.mp4',
            '2025-01-04_08-40-12.mp4'
        ]
    
    # Split data
    test_df = df[df['video_filename'].isin(test_videos)]
    train_val_df = df[~df['video_filename'].isin(test_videos)]
    
    # Further split train_val into train and validation
    train_videos = train_val_df['video_filename'].unique()
    train_videos, val_videos = train_test_split(
        train_videos, test_size=0.2, random_state=42
    )
    
    train_df = train_val_df[train_val_df['video_filename'].isin(train_videos)]
    val_df = train_val_df[train_val_df['video_filename'].isin(val_videos)]
    
    print(f"\nData split:")
    print(f"  Train: {len(train_df)} samples from {len(train_videos)} videos")
    print(f"  Val: {len(val_df)} samples from {len(val_videos)} videos")
    print(f"  Test: {len(test_df)} samples from {len(test_videos)} videos")
    
    # Identify feature columns (exclude metadata)
    metadata_cols = [
        'frame_filename', 'video_filename', 'frame_idx', 'labels',
        'left_action', 'left_outcome', 'right_action', 'right_outcome',
        'instance_id', 'width', 'height'
    ]
    
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Normalize features
    scaler = StandardScaler()
    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()
    
    # Fit scaler on training data
    scaler.fit(train_df[feature_cols])
    
    # Transform all sets
    train_df_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Save processed data
    base_name = "data_10hz_3d"
    if output_suffix:
        base_name += output_suffix
    output_dir = Path(f"input/{base_name}")
    output_dir.mkdir(exist_ok=True)
    
    train_df_scaled.to_csv(output_dir / "train_3d.csv", index=False)
    val_df_scaled.to_csv(output_dir / "val_3d.csv", index=False)
    test_df_scaled.to_csv(output_dir / "test_3d.csv", index=False)
    
    # Save scaler
    import joblib
    joblib.dump(scaler, output_dir / "scaler_3d.pkl")
    
    # Save feature information
    feature_info = {
        "feature_columns": feature_cols,
        "metadata_columns": metadata_cols,
        "num_features": len(feature_cols),
        "train_videos": list(train_videos),
        "val_videos": list(val_videos),
        "test_videos": test_videos
    }
    
    # Add filtering information to feature_info
    if depth_min is not None or depth_max is not None or volume_min is not None or volume_max is not None:
        feature_info["filtering_applied"] = {
            "depth_min": depth_min,
            "depth_max": depth_max,
            "volume_min": volume_min,
            "volume_max": volume_max,
            "only_2_instance_frames": True
        }
    
    with open(output_dir / "feature_info_3d.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\nSaved processed data to {output_dir}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Create a summary of filtering applied
    if depth_min is not None or depth_max is not None or volume_min is not None or volume_max is not None:
        summary_file = output_dir / "filtering_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("3D POSE DATA FILTERING SUMMARY\n")
            f.write("="*40 + "\n\n")
            f.write(f"Filtering parameters applied:\n")
            f.write(f"  Depth range: {depth_min} to {depth_max}\n")
            f.write(f"  Volume range: {volume_min} to {volume_max}\n")
            f.write(f"  Only frames with exactly 2 instances: Yes\n\n")
            f.write(f"Results:\n")
            f.write(f"  Train samples: {len(train_df_scaled)}\n")
            f.write(f"  Val samples: {len(val_df_scaled)}\n")
            f.write(f"  Test samples: {len(test_df_scaled)}\n")
            f.write(f"  Total features: {len(feature_cols)}\n")
        print(f"Filtering summary saved to {summary_file}")
    
    # Print sample features
    print("\nSample features:")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")
    
    return train_df_scaled, val_df_scaled, test_df_scaled, feature_cols


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Prepare 3D pose data for training with filtering")
    parser.add_argument(
        "--depth_min",
        type=float,
        default=None,
        help="Minimum depth threshold for filtering instances"
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=None,
        help="Maximum depth threshold for filtering instances"
    )
    parser.add_argument(
        "--volume_min",
        type=float,
        default=None,
        help="Minimum volume threshold for filtering instances"
    )
    parser.add_argument(
        "--volume_max",
        type=float,
        default=None,
        help="Maximum volume threshold for filtering instances"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="input/data_10hz",
        help="Directory containing 3D pose data"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix to add to output directory (e.g., '_filtered')"
    )
    
    args = parser.parse_args()
    
    # Load 3D pose data
    data_dir = Path(args.data_dir)
    pose_3d_path = data_dir / "pose_preds_3d.csv"
    
    if not pose_3d_path.exists():
        print(f"3D pose file not found at {pose_3d_path}")
        print("Please run process_pose_preds_3d.py first")
        return
    
    print("Loading 3D pose data...")
    pose_3d_df = pd.read_csv(pose_3d_path)
    print(f"Loaded {len(pose_3d_df)} instances from {len(pose_3d_df['video_filename'].unique())} videos")
    
    # Show filtering parameters
    print(f"\nFiltering parameters:")
    print(f"  Depth range: {args.depth_min} to {args.depth_max}")
    print(f"  Volume range: {args.volume_min} to {args.volume_max}")
    print(f"  Only frames with exactly 2 instances will be kept")
    
    # Prepare training data with filtering
    train_df, val_df, test_df, feature_cols = prepare_training_data(
        pose_3d_df, 
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        volume_min=args.volume_min,
        volume_max=args.volume_max,
        output_suffix=args.output_suffix
    )
    
    if train_df is None:
        print("Data preparation failed!")
        return
    
    print("\n=== Data Preparation Complete ===")
    print(f"Ready for training with {len(feature_cols)} 3D features!")
    
    # Show final statistics
    print(f"\nFinal dataset statistics:")
    print(f"  Train videos: {len(train_df['video_filename'].unique())}")
    print(f"  Val videos: {len(val_df['video_filename'].unique())}")
    print(f"  Test videos: {len(test_df['video_filename'].unique())}")
    
    # Show action distribution
    print(f"\nAction distribution in training data:")
    if 'left_action' in train_df.columns:
        left_actions = train_df['left_action'].value_counts()
        print("Left actions:")
        for action, count in left_actions.items():
            print(f"  {action}: {count}")
    
    if 'right_action' in train_df.columns:
        right_actions = train_df['right_action'].value_counts()
        print("Right actions:")
        for action, count in right_actions.items():
            print(f"  {action}: {count}")


if __name__ == "__main__":
    main()