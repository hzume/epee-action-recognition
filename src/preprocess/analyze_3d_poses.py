"""Analyze and visualize 3D pose predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Human3D keypoint connections for visualization
KEYPOINT_CONNECTIONS = [
    # Lower body connections
    (0, 1), (0, 4),  # pelvis to hips
    (1, 2), (2, 3),  # left hip to knee to ankle
    (4, 5), (5, 6),  # right hip to knee to ankle
    # Spine connections
    (0, 7), (7, 8), (8, 9), (9, 10),  # pelvis to spine to thorax to neck to head
    # Upper body connections
    (8, 11), (8, 14),  # thorax to shoulders
    (11, 12), (12, 13),  # left shoulder to elbow to wrist
    (14, 15), (15, 16),  # right shoulder to elbow to wrist
]

KEYPOINT_NAMES = [
    "pelvis", "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle", "spine",
    "thorax", "neck", "head", "left_shoulder",
    "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"
]


def load_pose_data():
    """Load 3D pose data"""
    data_dir = Path("input/data_10hz")
    
    # Check if processed files exist
    pose_3d_path = data_dir / "pose_preds_3d.csv"
    pose_3d_norm_path = data_dir / "pose_preds_3d_normalized.csv"
    
    if not pose_3d_path.exists():
        print(f"3D pose file not found at {pose_3d_path}")
        print("Please run process_pose_preds_3d.py first")
        return None, None
    
    print("Loading 3D pose data...")
    pose_3d_df = pd.read_csv(pose_3d_path)
    pose_3d_norm_df = pd.read_csv(pose_3d_norm_path) if pose_3d_norm_path.exists() else None
    
    return pose_3d_df, pose_3d_norm_df


def analyze_3d_statistics(pose_3d_df):
    """Analyze 3D pose statistics"""
    print("\n=== 3D Pose Statistics ===")
    
    # Analyze depth distribution
    z_coords = []
    for i in range(17):
        z_coords.extend(pose_3d_df[f"keypoint_{i}_z"].values)
    z_coords = np.array(z_coords)
    
    print(f"\nDepth (Z-axis) statistics:")
    print(f"  Mean: {z_coords.mean():.3f}")
    print(f"  Std: {z_coords.std():.3f}")
    print(f"  Min: {z_coords.min():.3f}")
    print(f"  Max: {z_coords.max():.3f}")
    
    # Analyze limb lengths
    print("\n=== Average Limb Lengths ===")
    limb_lengths = {}
    
    # Define limbs for Human3D keypoints
    limbs = {
        "upper_arm_left": (11, 12),   # left shoulder to elbow
        "upper_arm_right": (14, 15),  # right shoulder to elbow
        "forearm_left": (12, 13),     # left elbow to wrist
        "forearm_right": (15, 16),    # right elbow to wrist
        "thigh_left": (1, 2),         # left hip to knee
        "thigh_right": (4, 5),        # right hip to knee
        "shin_left": (2, 3),          # left knee to ankle
        "shin_right": (5, 6),         # right knee to ankle
        "torso_lower": (0, 7),        # pelvis to spine
        "torso_upper": (7, 8),        # spine to thorax
        "neck": (9, 10),              # neck to head
        "shoulder_span": (11, 14),    # left shoulder to right shoulder
        "hip_span": (1, 4),           # left hip to right hip
    }
    
    for limb_name, (kp1, kp2) in limbs.items():
        lengths = []
        for idx in pose_3d_df.index:
            p1 = np.array([
                pose_3d_df.loc[idx, f"keypoint_{kp1}_x"],
                pose_3d_df.loc[idx, f"keypoint_{kp1}_y"],
                pose_3d_df.loc[idx, f"keypoint_{kp1}_z"]
            ])
            p2 = np.array([
                pose_3d_df.loc[idx, f"keypoint_{kp2}_x"],
                pose_3d_df.loc[idx, f"keypoint_{kp2}_y"],
                pose_3d_df.loc[idx, f"keypoint_{kp2}_z"]
            ])
            length = np.linalg.norm(p2 - p1)
            lengths.append(length)
        
        limb_lengths[limb_name] = np.array(lengths)
        print(f"{limb_name:15s}: {np.mean(lengths):.3f} ± {np.std(lengths):.3f}")
    
    return limb_lengths


def visualize_3d_pose(pose_3d_df, frame_idx=0, instance_idx=0):
    """Visualize a single 3D pose"""
    # Get specific pose
    mask = (pose_3d_df['frame_idx'] == frame_idx) & (pose_3d_df['instance_id'] == instance_idx)
    if not mask.any():
        print(f"No data found for frame {frame_idx}, instance {instance_idx}")
        return
    
    row = pose_3d_df[mask].iloc[0]
    
    # Extract keypoints
    keypoints = np.array([
        [row[f"keypoint_{i}_x"], row[f"keypoint_{i}_y"], row[f"keypoint_{i}_z"]]
        for i in range(17)
    ])
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot keypoints
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='red', s=50)
    
    # Plot connections
    for connection in KEYPOINT_CONNECTIONS:
        kp1, kp2 = connection
        ax.plot([keypoints[kp1, 0], keypoints[kp2, 0]],
                [keypoints[kp1, 1], keypoints[kp2, 1]],
                [keypoints[kp1, 2], keypoints[kp2, 2]], 'b-', linewidth=2)
    
    # Add labels for some keypoints
    for i, name in enumerate(KEYPOINT_NAMES):
        if i in [0, 8, 9, 10, 11, 14]:  # Label key points: pelvis, thorax, neck, head, shoulders
            ax.text(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], name, fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title(f'3D Pose - Frame: {frame_idx}, Instance: {instance_idx}\n'
                 f'Video: {row["video_filename"]}')
    
    # Set equal aspect ratio
    max_range = np.array([
        keypoints[:, 0].max() - keypoints[:, 0].min(),
        keypoints[:, 1].max() - keypoints[:, 1].min(),
        keypoints[:, 2].max() - keypoints[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (keypoints[:, 0].max() + keypoints[:, 0].min()) * 0.5
    mid_y = (keypoints[:, 1].max() + keypoints[:, 1].min()) * 0.5
    mid_z = (keypoints[:, 2].max() + keypoints[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig


def analyze_pelvis_centered_coordinates(pose_3d_df):
    """Analyze pelvis-centered coordinate system properties"""
    print("\n=== Pelvis-Centered Coordinate Analysis ===")
    
    # Since pelvis is at origin (0) in human3d format, check if this is actually the case
    pelvis_coords = []
    for axis in ['x', 'y', 'z']:
        pelvis_coord = pose_3d_df[f'keypoint_0_{axis}'].values
        pelvis_coords.append(pelvis_coord)
        print(f"Pelvis {axis.upper()}-coordinate: mean={pelvis_coord.mean():.3f}, std={pelvis_coord.std():.3f}")
    
    # Check if poses are actually centered around pelvis
    pelvis_distances = np.sqrt(np.sum([coord**2 for coord in pelvis_coords], axis=0))
    print(f"Distance from pelvis to origin: mean={pelvis_distances.mean():.3f}, std={pelvis_distances.std():.3f}")
    
    # Analyze body orientation based on thorax-head direction
    if len(pose_3d_df) > 0:
        thorax_x, thorax_y, thorax_z = pose_3d_df['keypoint_8_x'], pose_3d_df['keypoint_8_y'], pose_3d_df['keypoint_8_z']
        head_x, head_y, head_z = pose_3d_df['keypoint_10_x'], pose_3d_df['keypoint_10_y'], pose_3d_df['keypoint_10_z']
        
        # Calculate upward direction vector
        up_x = head_x - thorax_x
        up_y = head_y - thorax_y  
        up_z = head_z - thorax_z
        
        # Calculate angles
        up_lengths = np.sqrt(up_x**2 + up_y**2 + up_z**2)
        valid_mask = up_lengths > 0
        
        if valid_mask.sum() > 0:
            # Vertical angle (tilt forward/backward)
            vertical_angles = np.arcsin(up_y[valid_mask] / up_lengths[valid_mask]) * 180 / np.pi
            print(f"\nBody vertical angles (degrees): mean={vertical_angles.mean():.1f}, std={vertical_angles.std():.1f}")
    
    return pelvis_distances


def analyze_3d_pose_quality(pose_3d_df):
    """Analyze 3D pose quality and consistency"""
    print("\n=== 3D Pose Quality Analysis ===")
    
    # Analyze keypoint score distribution
    print("\nKeypoint score statistics:")
    all_scores = []
    for i in range(17):
        score_col = f"keypoint_score_{i}"
        if score_col in pose_3d_df.columns:
            scores = pose_3d_df[score_col].values
            all_scores.extend(scores)
            print(f"  {KEYPOINT_NAMES[i]:15s}: mean={scores.mean():.3f}, std={scores.std():.3f}")
    
    if all_scores:
        all_scores = np.array(all_scores)
        print(f"\nOverall keypoint scores:")
        print(f"  Mean: {all_scores.mean():.3f}")
        print(f"  Std: {all_scores.std():.3f}")
        print(f"  Min: {all_scores.min():.3f}")
        print(f"  Max: {all_scores.max():.3f}")
    
    # Analyze pose stability over time
    print("\n=== Temporal Stability Analysis ===")
    
    stability_metrics = []
    for video in pose_3d_df['video_filename'].unique():
        video_df = pose_3d_df[pose_3d_df['video_filename'] == video]
        
        for instance_id in video_df['instance_id'].unique():
            instance_df = video_df[video_df['instance_id'] == instance_id].sort_values('frame_idx')
            
            if len(instance_df) < 2:
                continue
            
            # Calculate frame-to-frame displacement
            displacements = []
            for i in range(1, len(instance_df)):
                curr_pose = np.array([[instance_df.iloc[i][f"keypoint_{j}_x"], 
                                     instance_df.iloc[i][f"keypoint_{j}_y"], 
                                     instance_df.iloc[i][f"keypoint_{j}_z"]] for j in range(17)])
                prev_pose = np.array([[instance_df.iloc[i-1][f"keypoint_{j}_x"], 
                                     instance_df.iloc[i-1][f"keypoint_{j}_y"], 
                                     instance_df.iloc[i-1][f"keypoint_{j}_z"]] for j in range(17)])
                
                displacement = np.mean(np.linalg.norm(curr_pose - prev_pose, axis=1))
                displacements.append(displacement)
            
            if displacements:
                stability_metrics.append({
                    'video': video,
                    'instance': instance_id,
                    'mean_displacement': np.mean(displacements),
                    'std_displacement': np.std(displacements)
                })
    
    if stability_metrics:
        stability_df = pd.DataFrame(stability_metrics)
        print(f"Analyzed {len(stability_df)} pose sequences")
        print(f"Average frame-to-frame displacement: {stability_df['mean_displacement'].mean():.3f}")
        print(f"Displacement variability: {stability_df['std_displacement'].mean():.3f}")


def main():
    """Main analysis function"""
    # Load data
    pose_3d_df, pose_3d_norm_df = load_pose_data()
    
    if pose_3d_df is None:
        return
    
    # Analyze statistics
    limb_lengths = analyze_3d_statistics(pose_3d_df)
    
    # Analyze pelvis-centered coordinate system
    pelvis_distances = analyze_pelvis_centered_coordinates(pose_3d_df)
    
    # Analyze pose quality
    analyze_3d_pose_quality(pose_3d_df)
    
    # Visualize sample poses
    print("\n=== Visualizing Sample 3D Poses ===")
    
    # Get some sample frames with actions
    action_frames = pose_3d_df[
        (pose_3d_df['left_action'].notna()) | 
        (pose_3d_df['right_action'].notna())
    ].head(3)
    
    if len(action_frames) == 0:
        # If no action frames, get first few frames
        action_frames = pose_3d_df.head(3)
    
    for idx, row in action_frames.iterrows():
        fig = visualize_3d_pose(pose_3d_df, row['frame_idx'], row['instance_id'])
        if fig is not None:
            output_path = f"output/3d_pose_sample_{idx}.png"
            fig.savefig(output_path, dpi=150)
            print(f"Saved visualization to {output_path}")
            plt.close(fig)
    
    # Plot depth distribution
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different analyses
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Depth distribution
    z_coords = []
    for i in range(17):
        z_coords.extend(pose_3d_df[f"keypoint_{i}_z"].values)
    
    ax1.hist(z_coords, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Depth (Z coordinate)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of 3D Pose Depths')
    ax1.grid(True, alpha=0.3)
    
    # 2. Limb length distribution (torso_upper as example)
    if 'torso_upper' in limb_lengths:
        ax2.hist(limb_lengths['torso_upper'], bins=30, alpha=0.7, edgecolor='black', color='green')
        ax2.set_xlabel('Upper Torso Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Upper Torso Lengths')
        ax2.grid(True, alpha=0.3)
    elif 'torso_lower' in limb_lengths:
        ax2.hist(limb_lengths['torso_lower'], bins=30, alpha=0.7, edgecolor='black', color='green')
        ax2.set_xlabel('Lower Torso Length')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Lower Torso Lengths')
        ax2.grid(True, alpha=0.3)
    
    # 3. X-Y coordinate distribution (pelvis position)
    if len(pose_3d_df) > 0:
        pelvis_x = pose_3d_df['keypoint_0_x']  # pelvis x
        pelvis_y = pose_3d_df['keypoint_0_y']  # pelvis y
        ax3.scatter(pelvis_x, pelvis_y, alpha=0.5, s=1)
        ax3.set_xlabel('Pelvis X')
        ax3.set_ylabel('Pelvis Y')
        ax3.set_title('Pelvis Positions (X-Y plane)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Keypoint score distribution
    score_cols = [f"keypoint_score_{i}" for i in range(17) if f"keypoint_score_{i}" in pose_3d_df.columns]
    if score_cols:
        all_scores = []
        for col in score_cols:
            all_scores.extend(pose_3d_df[col].values)
        
        ax4.hist(all_scores, bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax4.set_xlabel('Keypoint Confidence Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Keypoint Confidence Scores')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/3d_pose_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved comprehensive 3D pose analysis to output/3d_pose_analysis.png")
    plt.close()
    
    # Analyze normalized poses if available
    if pose_3d_norm_df is not None:
        print("\n=== Normalized 3D Pose Analysis ===")
        print(f"Normalized poses: {len(pose_3d_norm_df)} samples")
        
        # Compare range of normalized vs original poses
        print("\nCoordinate ranges comparison:")
        for axis in ['x', 'y', 'z']:
            orig_coords = []
            norm_coords = []
            for i in range(17):
                orig_coords.extend(pose_3d_df[f"keypoint_{i}_{axis}"].values)
                norm_coords.extend(pose_3d_norm_df[f"keypoint_{i}_{axis}"].values)
            
            orig_coords = np.array(orig_coords)
            norm_coords = np.array(norm_coords)
            
            print(f"  {axis.upper()}-axis:")
            print(f"    Original: [{orig_coords.min():.3f}, {orig_coords.max():.3f}]")
            print(f"    Normalized: [{norm_coords.min():.3f}, {norm_coords.max():.3f}]")
        
        # Analyze pelvis position in both datasets
        print("\nPelvis position analysis:")
        for dataset_name, df in [("Original", pose_3d_df), ("Normalized", pose_3d_norm_df)]:
            pelvis_x = df['keypoint_0_x'].values
            pelvis_y = df['keypoint_0_y'].values
            pelvis_z = df['keypoint_0_z'].values
            pelvis_mean = [pelvis_x.mean(), pelvis_y.mean(), pelvis_z.mean()]
            pelvis_std = [pelvis_x.std(), pelvis_y.std(), pelvis_z.std()]
            print(f"  {dataset_name} pelvis center: [{pelvis_mean[0]:.3f}, {pelvis_mean[1]:.3f}, {pelvis_mean[2]:.3f}]")
            print(f"  {dataset_name} pelvis std: [{pelvis_std[0]:.3f}, {pelvis_std[1]:.3f}, {pelvis_std[2]:.3f}]")
    
    print("\n=== 3D Pose Analysis Complete ===")


if __name__ == "__main__":
    main()