"""Prepare 3D pose data for training by creating appropriate data splits and features"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    
    # Define joints for angle calculation
    angle_joints = [
        ("left_elbow_angle", 5, 7, 9),      # shoulder-elbow-wrist
        ("right_elbow_angle", 6, 8, 10),
        ("left_knee_angle", 11, 13, 15),    # hip-knee-ankle
        ("right_knee_angle", 12, 14, 16),
        ("left_shoulder_angle", 11, 5, 7),  # hip-shoulder-elbow
        ("right_shoulder_angle", 12, 6, 8),
        ("left_hip_angle", 5, 11, 13),      # shoulder-hip-knee
        ("right_hip_angle", 6, 12, 14),
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
    
    # 3. Add distance features between players
    print("Adding inter-player distance features...")
    
    # For frames with exactly 2 instances (2 players)
    two_player_frames = df.groupby('frame_filename').filter(lambda x: len(x) == 2)
    
    if len(two_player_frames) > 0:
        distance_features = []
        
        for frame in two_player_frames['frame_filename'].unique():
            frame_data = two_player_frames[two_player_frames['frame_filename'] == frame]
            if len(frame_data) == 2:
                player1 = frame_data.iloc[0]
                player2 = frame_data.iloc[1]
                
                # Calculate distances between corresponding keypoints
                distances = {}
                for kp in range(17):
                    p1 = np.array([player1[f"keypoint_{kp}_x"], 
                                  player1[f"keypoint_{kp}_y"], 
                                  player1[f"keypoint_{kp}_z"]])
                    p2 = np.array([player2[f"keypoint_{kp}_x"], 
                                  player2[f"keypoint_{kp}_y"], 
                                  player2[f"keypoint_{kp}_z"]])
                    distances[f"inter_player_dist_{kp}"] = np.linalg.norm(p2 - p1)
                
                # Add to both players
                for idx in frame_data.index:
                    for key, value in distances.items():
                        df.loc[idx, key] = value
    
    # 4. Add body orientation features
    print("Adding body orientation features...")
    
    # Calculate torso vector (from hip center to shoulder center)
    for idx in df.index:
        # Hip center
        hip_center = np.array([
            (df.loc[idx, "keypoint_11_x"] + df.loc[idx, "keypoint_12_x"]) / 2,
            (df.loc[idx, "keypoint_11_y"] + df.loc[idx, "keypoint_12_y"]) / 2,
            (df.loc[idx, "keypoint_11_z"] + df.loc[idx, "keypoint_12_z"]) / 2
        ])
        
        # Shoulder center
        shoulder_center = np.array([
            (df.loc[idx, "keypoint_5_x"] + df.loc[idx, "keypoint_6_x"]) / 2,
            (df.loc[idx, "keypoint_5_y"] + df.loc[idx, "keypoint_6_y"]) / 2,
            (df.loc[idx, "keypoint_5_z"] + df.loc[idx, "keypoint_6_z"]) / 2
        ])
        
        # Torso vector
        torso_vector = shoulder_center - hip_center
        torso_length = np.linalg.norm(torso_vector)
        
        if torso_length > 0:
            torso_vector = torso_vector / torso_length
            
            # Store orientation angles
            df.loc[idx, "torso_pitch"] = np.arcsin(torso_vector[1])  # Vertical tilt
            df.loc[idx, "torso_yaw"] = np.arctan2(torso_vector[0], torso_vector[2])  # Horizontal rotation
    
    return df


def prepare_training_data(pose_3d_df, test_videos=None):
    """Prepare data for training with proper splits"""
    
    # Create enhanced features
    df = create_3d_features(pose_3d_df)
    
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
    output_dir = Path("input/data_10hz_3d")
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
    
    with open(output_dir / "feature_info_3d.json", "w") as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\nSaved processed data to {output_dir}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Print sample features
    print("\nSample features:")
    for i, col in enumerate(feature_cols[:10]):
        print(f"  {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")
    
    return train_df_scaled, val_df_scaled, test_df_scaled, feature_cols


def main():
    """Main function"""
    # Load 3D pose data
    data_dir = Path("input/data_10hz")
    pose_3d_path = data_dir / "pose_preds_3d.csv"
    
    if not pose_3d_path.exists():
        print(f"3D pose file not found at {pose_3d_path}")
        print("Please run process_pose_preds_3d.py first")
        return
    
    print("Loading 3D pose data...")
    pose_3d_df = pd.read_csv(pose_3d_path)
    
    # Prepare training data
    train_df, val_df, test_df, feature_cols = prepare_training_data(pose_3d_df)
    
    print("\n=== Data Preparation Complete ===")
    print(f"Ready for training with {len(feature_cols)} 3D features!")


if __name__ == "__main__":
    main()