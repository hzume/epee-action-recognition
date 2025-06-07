"""Analyze and visualize 3D pose predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# COCO keypoint connections for visualization
KEYPOINT_CONNECTIONS = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Lower body
    (11, 13), (13, 15), (12, 14), (14, 16)
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def load_pose_data():
    """Load both 2D and 3D pose data"""
    data_dir = Path("input/data_10hz")
    
    # Check if processed files exist
    pose_2d_path = data_dir / "pose_preds.csv"
    pose_3d_path = data_dir / "pose_preds_3d.csv"
    pose_3d_norm_path = data_dir / "pose_preds_3d_normalized.csv"
    
    if not pose_3d_path.exists():
        print(f"3D pose file not found at {pose_3d_path}")
        print("Please run process_pose_preds_3d.py first")
        return None, None, None
    
    print("Loading pose data...")
    pose_2d_df = pd.read_csv(pose_2d_path) if pose_2d_path.exists() else None
    pose_3d_df = pd.read_csv(pose_3d_path)
    pose_3d_norm_df = pd.read_csv(pose_3d_norm_path) if pose_3d_norm_path.exists() else None
    
    return pose_2d_df, pose_3d_df, pose_3d_norm_df


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
    
    # Define limbs
    limbs = {
        "upper_arm_left": (5, 7),    # left shoulder to elbow
        "upper_arm_right": (6, 8),   # right shoulder to elbow
        "forearm_left": (7, 9),      # left elbow to wrist
        "forearm_right": (8, 10),    # right elbow to wrist
        "thigh_left": (11, 13),      # left hip to knee
        "thigh_right": (12, 14),     # right hip to knee
        "shin_left": (13, 15),       # left knee to ankle
        "shin_right": (14, 16),      # right knee to ankle
        "torso": (11, 5),            # left hip to left shoulder
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
        if i in [0, 5, 6, 11, 12]:  # Label key points only
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


def compare_2d_3d_consistency(pose_2d_df, pose_3d_df):
    """Compare consistency between 2D and 3D poses"""
    if pose_2d_df is None:
        print("2D pose data not available for comparison")
        return
    
    print("\n=== 2D vs 3D Consistency Analysis ===")
    
    # Merge dataframes on common keys
    merge_keys = ['frame_filename', 'instance_id']
    merged_df = pd.merge(
        pose_2d_df[merge_keys + [f"keypoint_{i}_x" for i in range(17)] + [f"keypoint_{i}_y" for i in range(17)]],
        pose_3d_df[merge_keys + [f"keypoint_{i}_x" for i in range(17)] + [f"keypoint_{i}_y" for i in range(17)] + [f"keypoint_{i}_z" for i in range(17)]],
        on=merge_keys,
        suffixes=('_2d', '_3d')
    )
    
    print(f"Matched {len(merged_df)} instances for comparison")
    
    # Calculate projection error (assuming 3D x,y should roughly match 2D after scaling)
    errors = []
    for i in range(17):
        x_diff = merged_df[f"keypoint_{i}_x_2d"] - merged_df[f"keypoint_{i}_x_3d"]
        y_diff = merged_df[f"keypoint_{i}_y_2d"] - merged_df[f"keypoint_{i}_y_3d"]
        error = np.sqrt(x_diff**2 + y_diff**2)
        errors.append(error.mean())
        
    print("\nAverage projection error per keypoint:")
    for i, (name, error) in enumerate(zip(KEYPOINT_NAMES, errors)):
        print(f"  {name:15s}: {error:.2f} pixels")


def main():
    """Main analysis function"""
    # Load data
    pose_2d_df, pose_3d_df, pose_3d_norm_df = load_pose_data()
    
    if pose_3d_df is None:
        return
    
    # Analyze statistics
    limb_lengths = analyze_3d_statistics(pose_3d_df)
    
    # Visualize sample poses
    print("\n=== Visualizing Sample 3D Poses ===")
    
    # Get some sample frames with actions
    action_frames = pose_3d_df[
        (pose_3d_df['left_action'].notna()) | 
        (pose_3d_df['right_action'].notna())
    ].head(3)
    
    for idx, row in action_frames.iterrows():
        fig = visualize_3d_pose(pose_3d_df, row['frame_idx'], row['instance_id'])
        output_path = f"output/3d_pose_sample_{idx}.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
        plt.close(fig)
    
    # Compare with 2D if available
    if pose_2d_df is not None:
        compare_2d_3d_consistency(pose_2d_df, pose_3d_df)
    
    # Plot depth distribution
    plt.figure(figsize=(10, 6))
    z_coords = []
    for i in range(17):
        z_coords.extend(pose_3d_df[f"keypoint_{i}_z"].values)
    
    plt.hist(z_coords, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Depth (Z coordinate)')
    plt.ylabel('Frequency')
    plt.title('Distribution of 3D Pose Depths')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output/3d_depth_distribution.png', dpi=150)
    print("Saved depth distribution plot to output/3d_depth_distribution.png")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()