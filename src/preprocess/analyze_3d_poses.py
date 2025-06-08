"""Analyze and visualize 3D pose predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import argparse
from scipy.optimize import minimize
from itertools import product

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


def analyze_depth_distribution(pose_3d_df, depth_threshold=None):
    """Analyze depth distribution and create visualizations"""
    print("\n=== Depth Distribution Analysis ===")
    
    # Calculate per-instance depth (using pelvis depth as representative)
    instance_depths = []
    for _, row in pose_3d_df.iterrows():
        pelvis_depth = row['keypoint_0_z']  # pelvis depth
        instance_depths.append({
            'video_filename': row['video_filename'],
            'frame_idx': row['frame_idx'],
            'instance_id': row['instance_id'],
            'depth': pelvis_depth
        })
    
    depth_df = pd.DataFrame(instance_depths)
    depths = depth_df['depth'].values
    
    print(f"Instance depth statistics (using pelvis):")
    print(f"  Count: {len(depths)}")
    print(f"  Mean: {depths.mean():.3f}")
    print(f"  Std: {depths.std():.3f}")
    print(f"  Min: {depths.min():.3f}")
    print(f"  Max: {depths.max():.3f}")
    print(f"  25th percentile: {np.percentile(depths, 25):.3f}")
    print(f"  50th percentile (median): {np.percentile(depths, 50):.3f}")
    print(f"  75th percentile: {np.percentile(depths, 75):.3f}")
    
    # Plot depth distribution
    plt.figure(figsize=(12, 8))
    
    # Create 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall depth histogram
    ax1.hist(depths, bins=50, alpha=0.7, edgecolor='black', color='blue')
    ax1.set_xlabel('Depth (Z coordinate)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Instance Depths (Pelvis)')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(depths.mean(), color='red', linestyle='--', label=f'Mean: {depths.mean():.3f}')
    ax1.axvline(np.median(depths), color='green', linestyle='--', label=f'Median: {np.median(depths):.3f}')
    if depth_threshold is not None:
        ax1.axvline(depth_threshold, color='orange', linestyle='-', linewidth=2, label=f'Threshold: {depth_threshold:.3f}')
    ax1.legend()
    
    # 2. Depth by video
    videos = depth_df['video_filename'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(videos)))
    for i, video in enumerate(videos):
        video_depths = depth_df[depth_df['video_filename'] == video]['depth']
        ax2.hist(video_depths, bins=30, alpha=0.5, label=video[:20], color=colors[i])
    ax2.set_xlabel('Depth (Z coordinate)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Depth Distribution by Video')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # 3. Depth vs frame index
    ax3.scatter(depth_df['frame_idx'], depth_df['depth'], alpha=0.5, s=1)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Depth')
    ax3.set_title('Depth vs Frame Index')
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot by video
    video_depth_data = [depth_df[depth_df['video_filename'] == video]['depth'].values for video in videos]
    ax4.boxplot(video_depth_data, labels=[v[:15] for v in videos])
    ax4.set_ylabel('Depth')
    ax4.set_title('Depth Distribution by Video (Box Plot)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/depth_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved depth distribution analysis to output/depth_distribution_analysis.png")
    plt.close()
    
    return depth_df


def analyze_bbox_volume_distribution(pose_3d_df, volume_min=None, volume_max=None):
    """Analyze bounding box volume distribution and create visualizations"""
    print("\n=== Bounding Box Volume Distribution Analysis ===")
    
    # Calculate per-instance bbox volume
    instance_volumes = []
    for idx, row in pose_3d_df.iterrows():
        volume, width, height, depth = calculate_bbox_volume(row)
        instance_volumes.append({
            'video_filename': row['video_filename'],
            'frame_idx': row['frame_idx'],
            'instance_id': row['instance_id'],
            'volume': volume,
            'width': width,
            'height': height,
            'depth': depth
        })
    
    volume_df = pd.DataFrame(instance_volumes)
    volumes = volume_df['volume'].values
    
    print(f"Bounding box volume statistics:")
    print(f"  Count: {len(volumes)}")
    print(f"  Mean: {volumes.mean():.3f}")
    print(f"  Std: {volumes.std():.3f}")
    print(f"  Min: {volumes.min():.3f}")
    print(f"  Max: {volumes.max():.3f}")
    print(f"  25th percentile: {np.percentile(volumes, 25):.3f}")
    print(f"  50th percentile (median): {np.percentile(volumes, 50):.3f}")
    print(f"  75th percentile: {np.percentile(volumes, 75):.3f}")
    
    # Plot volume distribution
    plt.figure(figsize=(12, 8))
    
    # Create 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Overall volume histogram
    ax1.hist(volumes, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax1.set_xlabel('Bounding Box Volume')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of 3D Bounding Box Volumes')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(volumes.mean(), color='red', linestyle='--', label=f'Mean: {volumes.mean():.3f}')
    ax1.axvline(np.median(volumes), color='green', linestyle='--', label=f'Median: {np.median(volumes):.3f}')
    if volume_min is not None:
        ax1.axvline(volume_min, color='orange', linestyle='-', linewidth=2, label=f'Min: {volume_min:.3f}')
    if volume_max is not None:
        ax1.axvline(volume_max, color='red', linestyle='-', linewidth=2, label=f'Max: {volume_max:.3f}')
    ax1.legend()
    
    # 2. Volume by video
    videos = volume_df['video_filename'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(videos)))
    for i, video in enumerate(videos):
        video_volumes = volume_df[volume_df['video_filename'] == video]['volume']
        ax2.hist(video_volumes, bins=30, alpha=0.5, label=video[:20], color=colors[i])
    ax2.set_xlabel('Bounding Box Volume')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Volume Distribution by Video')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # 3. Volume vs frame index
    ax3.scatter(volume_df['frame_idx'], volume_df['volume'], alpha=0.5, s=1)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Volume')
    ax3.set_title('Bounding Box Volume vs Frame Index')
    ax3.grid(True, alpha=0.3)
    
    # 4. Box dimensions breakdown
    ax4.boxplot([volume_df['width'].values, volume_df['height'].values, volume_df['depth'].values],
                labels=['Width', 'Height', 'Depth'])
    ax4.set_ylabel('Dimension Size')
    ax4.set_title('Bounding Box Dimensions Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/bbox_volume_distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved bbox volume distribution analysis to output/bbox_volume_distribution_analysis.png")
    plt.close()
    
    return volume_df


def analyze_instances_per_frame(pose_3d_df, depth_threshold=None, volume_min=None, volume_max=None):
    """Analyze number of instances per frame, with optional depth and volume filtering"""
    print(f"\n=== Instances Per Frame Analysis ===")
    
    df_to_analyze = pose_3d_df.copy()
    original_count = len(df_to_analyze)
    
    # Apply depth filtering if threshold is provided
    if depth_threshold is not None:
        print(f"Applying depth threshold: {depth_threshold}")
        # Filter by pelvis depth
        df_to_analyze = df_to_analyze[df_to_analyze['keypoint_0_z'] <= depth_threshold]
        depth_filtered_count = len(df_to_analyze)
        print(f"Instances after depth filtering: {depth_filtered_count}/{original_count} ({depth_filtered_count/original_count*100:.1f}%)")
    
    # Apply volume filtering if range is provided
    if volume_min is not None or volume_max is not None:
        print(f"Applying volume filtering: min={volume_min}, max={volume_max}")
        pre_volume_count = len(df_to_analyze)
        
        # Calculate volumes for remaining instances
        volumes = []
        valid_indices = []
        for idx, row in df_to_analyze.iterrows():
            volume, _, _, _ = calculate_bbox_volume(row)
            
            # Check volume range
            volume_valid = True
            if volume_min is not None and volume < volume_min:
                volume_valid = False
            if volume_max is not None and volume > volume_max:
                volume_valid = False
            
            if volume_valid:
                volumes.append(volume)
                valid_indices.append(idx)
        
        # Filter by volume
        df_to_analyze = df_to_analyze.loc[valid_indices]
        volume_filtered_count = len(df_to_analyze)
        print(f"Instances after volume filtering: {volume_filtered_count}/{pre_volume_count} ({volume_filtered_count/pre_volume_count*100:.1f}%)")
    
    if depth_threshold is None and volume_min is None and volume_max is None:
        print("No filtering applied")
    
    # Count instances per frame
    frame_instance_counts = []
    
    for video in df_to_analyze['video_filename'].unique():
        video_df = df_to_analyze[df_to_analyze['video_filename'] == video]
        
        for frame_idx in video_df['frame_idx'].unique():
            frame_df = video_df[video_df['frame_idx'] == frame_idx]
            instance_count = len(frame_df)
            
            frame_instance_counts.append({
                'video_filename': video,
                'frame_idx': frame_idx,
                'instance_count': instance_count
            })
    
    instances_df = pd.DataFrame(frame_instance_counts)
    counts = instances_df['instance_count'].values
    
    print(f"\nInstances per frame statistics:")
    print(f"  Total frames analyzed: {len(instances_df)}")
    print(f"  Mean instances per frame: {counts.mean():.2f}")
    print(f"  Std: {counts.std():.2f}")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  Median: {np.median(counts):.1f}")
    
    # Count distribution
    print(f"\nInstance count distribution:")
    unique_counts, count_freq = np.unique(counts, return_counts=True)
    for count, freq in zip(unique_counts, count_freq):
        print(f"  {count} instances: {freq} frames ({freq/len(counts)*100:.1f}%)")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Histogram of instances per frame
    ax1.hist(counts, bins=range(int(counts.min()), int(counts.max())+2), 
             alpha=0.7, edgecolor='black', color='green', align='left')
    ax1.set_xlabel('Number of Instances per Frame')
    ax1.set_ylabel('Number of Frames')
    title = 'Distribution of Instances per Frame'
    filters = []
    if depth_threshold is not None:
        filters.append(f'Depth ≤ {depth_threshold:.2f}')
    if volume_min is not None and volume_max is not None:
        filters.append(f'Volume: {volume_min:.2f}-{volume_max:.2f}')
    elif volume_min is not None:
        filters.append(f'Volume ≥ {volume_min:.2f}')
    elif volume_max is not None:
        filters.append(f'Volume ≤ {volume_max:.2f}')
    
    if filters:
        title += f' ({", ".join(filters)})'
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(int(counts.min()), int(counts.max())+1))
    
    # Add statistics text
    stats_text = f'Mean: {counts.mean():.2f}\nStd: {counts.std():.2f}\nTotal frames: {len(counts)}'
    ax1.text(0.7, 0.8, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    # 2. Instances per frame by video
    videos = instances_df['video_filename'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(videos)))
    
    for i, video in enumerate(videos):
        video_counts = instances_df[instances_df['video_filename'] == video]['instance_count']
        ax2.hist(video_counts, bins=range(int(counts.min()), int(counts.max())+2),
                alpha=0.6, label=video[:20], color=colors[i], align='left')
    
    ax2.set_xlabel('Number of Instances per Frame')
    ax2.set_ylabel('Number of Frames')
    ax2.set_title('Instances per Frame by Video')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_xticks(range(int(counts.min()), int(counts.max())+1))
    
    plt.tight_layout()
    
    # Save with appropriate filename based on applied filters
    filename_parts = ['output/instances_per_frame']
    if depth_threshold is not None:
        filename_parts.append(f'depth_{depth_threshold:.2f}')
    if volume_min is not None and volume_max is not None:
        filename_parts.append(f'volume_{volume_min:.2f}-{volume_max:.2f}')
    elif volume_min is not None:
        filename_parts.append(f'volume_min_{volume_min:.2f}')
    elif volume_max is not None:
        filename_parts.append(f'volume_max_{volume_max:.2f}')
    
    if len(filename_parts) == 1:
        filename = 'output/instances_per_frame_analysis.png'
    else:
        filename = '_'.join(filename_parts) + '.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved instances per frame analysis to {filename}")
    plt.close()
    
    return instances_df


def count_2instance_frames(pose_3d_df, depth_min=None, depth_max=None, volume_min=None, volume_max=None):
    """Count the number of frames that have exactly 2 instances after filtering"""
    df_filtered = pose_3d_df.copy()
    
    # Apply depth filtering
    if depth_min is not None:
        df_filtered = df_filtered[df_filtered['keypoint_0_z'] >= depth_min]
    if depth_max is not None:
        df_filtered = df_filtered[df_filtered['keypoint_0_z'] <= depth_max]
    
    # Apply volume filtering
    if volume_min is not None or volume_max is not None:
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
        
        df_filtered = df_filtered.loc[valid_indices]
    
    # Count instances per frame
    frame_counts = {}
    for _, row in df_filtered.iterrows():
        video_frame_key = (row['video_filename'], row['frame_idx'])
        frame_counts[video_frame_key] = frame_counts.get(video_frame_key, 0) + 1
    
    # Count frames with exactly 2 instances
    frames_with_2_instances = sum(1 for count in frame_counts.values() if count == 2)
    total_frames = len(frame_counts)
    
    return frames_with_2_instances, total_frames


def optimize_for_2_instances(pose_3d_df, method='grid_search'):
    """Optimize depth and volume thresholds to maximize frames with exactly 2 instances
    
    Args:
        pose_3d_df: 3D pose dataframe
        method: 'grid_search' or 'scipy' optimization method
        
    Returns:
        dict: Optimal parameters and results
    """
    print("\n=== Optimizing Parameters for Maximum 2-Instance Frames ===")
    
    # Calculate initial statistics
    initial_2instances, initial_total = count_2instance_frames(pose_3d_df)
    print(f"Initial (no filtering): {initial_2instances}/{initial_total} frames with 2 instances ({initial_2instances/initial_total*100:.1f}%)")
    
    # Calculate ranges for optimization based on data distribution
    depths = pose_3d_df['keypoint_0_z'].values
    depth_min_range = (depths.min(), np.percentile(depths, 75))
    depth_max_range = (np.percentile(depths, 25), depths.max())
    
    # Calculate volume ranges
    volumes = []
    print("Calculating volume ranges...")
    for idx, row in pose_3d_df.iterrows():
        volume, _, _, _ = calculate_bbox_volume(row)
        volumes.append(volume)
    volumes = np.array(volumes)
    
    volume_min_range = (volumes.min(), np.percentile(volumes, 75))
    volume_max_range = (np.percentile(volumes, 25), volumes.max())
    
    print(f"Search ranges:")
    print(f"  Depth min: {depth_min_range[0]:.3f} to {depth_min_range[1]:.3f}")
    print(f"  Depth max: {depth_max_range[0]:.3f} to {depth_max_range[1]:.3f}")
    print(f"  Volume min: {volume_min_range[0]:.3f} to {volume_min_range[1]:.3f}")
    print(f"  Volume max: {volume_max_range[0]:.3f} to {volume_max_range[1]:.3f}")
    
    if method == 'grid_search':
        return _grid_search_optimization(pose_3d_df, depth_min_range, depth_max_range, 
                                       volume_min_range, volume_max_range)
    elif method == 'scipy':
        return _scipy_optimization(pose_3d_df, depth_min_range, depth_max_range,
                                 volume_min_range, volume_max_range)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _grid_search_optimization(pose_3d_df, depth_min_range, depth_max_range, volume_min_range, volume_max_range):
    """Grid search optimization for maximum 2-instance frames"""
    print("\nRunning grid search optimization...")
    
    # Create parameter grids
    depth_min_vals = np.linspace(depth_min_range[0], depth_min_range[1], 10)
    depth_max_vals = np.linspace(depth_max_range[0], depth_max_range[1], 10)
    volume_min_vals = np.linspace(volume_min_range[0], volume_min_range[1], 8)
    volume_max_vals = np.linspace(volume_max_range[0], volume_max_range[1], 8)
    
    best_score = -1
    best_params = None
    best_results = None
    
    total_combinations = len(depth_min_vals) * len(depth_max_vals) * len(volume_min_vals) * len(volume_max_vals)
    print(f"Testing {total_combinations} parameter combinations...")
    
    eval_count = 0
    
    for depth_min in depth_min_vals:
        for depth_max in depth_max_vals:
            # Skip invalid depth ranges
            if depth_min >= depth_max:
                continue
                
            for volume_min in volume_min_vals:
                for volume_max in volume_max_vals:
                    # Skip invalid volume ranges
                    if volume_min >= volume_max:
                        continue
                    
                    eval_count += 1
                    if eval_count % 100 == 0:
                        print(f"  Progress: {eval_count}/{total_combinations} ({eval_count/total_combinations*100:.1f}%)")
                    
                    # Count 2-instance frames with these parameters
                    frames_2instances, total_frames = count_2instance_frames(
                        pose_3d_df, depth_min, depth_max, volume_min, volume_max
                    )
                    
                    if total_frames == 0:
                        continue
                    
                    # Score: percentage of frames with exactly 2 instances
                    score = frames_2instances / total_frames
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'depth_min': depth_min,
                            'depth_max': depth_max,
                            'volume_min': volume_min,
                            'volume_max': volume_max
                        }
                        best_results = {
                            'frames_with_2instances': frames_2instances,
                            'total_frames': total_frames,
                            'percentage': score * 100
                        }
    
    print(f"\nGrid search completed. Evaluated {eval_count} valid combinations.")
    return best_params, best_results


def _scipy_optimization(pose_3d_df, depth_min_range, depth_max_range, volume_min_range, volume_max_range):
    """SciPy-based optimization for maximum 2-instance frames"""
    print("\nRunning SciPy optimization...")
    
    def objective(params):
        depth_min, depth_max, volume_min, volume_max = params
        
        # Skip invalid ranges
        if depth_min >= depth_max or volume_min >= volume_max:
            return -1e6  # Large penalty
        
        frames_2instances, total_frames = count_2instance_frames(
            pose_3d_df, depth_min, depth_max, volume_min, volume_max
        )
        
        if total_frames == 0:
            return -1e6  # Large penalty
        
        # Return negative percentage (since minimize finds minimum)
        return -(frames_2instances / total_frames)
    
    # Initial guess (middle of ranges)
    x0 = [
        (depth_min_range[0] + depth_min_range[1]) / 2,
        (depth_max_range[0] + depth_max_range[1]) / 2,
        (volume_min_range[0] + volume_min_range[1]) / 2,
        (volume_max_range[0] + volume_max_range[1]) / 2
    ]
    
    # Bounds
    bounds = [
        depth_min_range,
        depth_max_range,
        volume_min_range,
        volume_max_range
    ]
    
    # Constraints to ensure valid ranges
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.01},  # depth_max > depth_min + small margin
        {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.001}  # volume_max > volume_min + small margin
    ]
    
    # Multiple optimization attempts with different methods
    best_result = None
    best_score = -1e6
    
    for method in ['L-BFGS-B', 'SLSQP', 'TNC']:
        try:
            result = minimize(
                objective, x0, method=method, bounds=bounds,
                constraints=constraints if method != 'L-BFGS-B' else None,
                options={'maxiter': 200}
            )
            
            if result.success and result.fun < best_score:
                best_result = result
                best_score = result.fun
        except Exception as e:
            print(f"  Method {method} failed: {e}")
            continue
    
    if best_result is None:
        print("SciPy optimization failed, falling back to grid search")
        return _grid_search_optimization(pose_3d_df, depth_min_range, depth_max_range,
                                       volume_min_range, volume_max_range)
    
    # Extract best parameters
    depth_min, depth_max, volume_min, volume_max = best_result.x
    frames_2instances, total_frames = count_2instance_frames(
        pose_3d_df, depth_min, depth_max, volume_min, volume_max
    )
    
    best_params = {
        'depth_min': depth_min,
        'depth_max': depth_max,
        'volume_min': volume_min,
        'volume_max': volume_max
    }
    
    best_results = {
        'frames_with_2instances': frames_2instances,
        'total_frames': total_frames,
        'percentage': (frames_2instances / total_frames) * 100 if total_frames > 0 else 0
    }
    
    return best_params, best_results


def run_optimization_analysis(pose_3d_df, method='grid_search'):
    """Run optimization analysis and display results"""
    print("\n" + "="*60)
    print("OPTIMIZATION ANALYSIS: MAXIMIZE 2-INSTANCE FRAMES")
    print("="*60)
    
    # Run optimization
    best_params, best_results = optimize_for_2_instances(pose_3d_df, method=method)
    
    if best_params is None:
        print("Optimization failed - no valid parameters found")
        return None, None
    
    print(f"\n🎯 OPTIMAL PARAMETERS FOUND:")
    print(f"  Depth range: {best_params['depth_min']:.3f} to {best_params['depth_max']:.3f}")
    print(f"  Volume range: {best_params['volume_min']:.3f} to {best_params['volume_max']:.3f}")
    
    print(f"\n📊 OPTIMIZATION RESULTS:")
    print(f"  Frames with exactly 2 instances: {best_results['frames_with_2instances']}")
    print(f"  Total frames after filtering: {best_results['total_frames']}")
    print(f"  Percentage: {best_results['percentage']:.1f}%")
    
    # Compare with no filtering
    initial_2instances, initial_total = count_2instance_frames(pose_3d_df)
    initial_percentage = (initial_2instances / initial_total) * 100
    improvement = best_results['percentage'] - initial_percentage
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Before optimization: {initial_2instances}/{initial_total} ({initial_percentage:.1f}%)")
    print(f"  After optimization: {best_results['frames_with_2instances']}/{best_results['total_frames']} ({best_results['percentage']:.1f}%)")
    print(f"  Improvement: +{improvement:.1f} percentage points")
    
    # Analyze the optimized dataset
    print(f"\n📋 ANALYSIS WITH OPTIMAL PARAMETERS:")
    
    # Apply optimal filtering and analyze
    df_filtered = pose_3d_df.copy()
    df_filtered = df_filtered[
        (df_filtered['keypoint_0_z'] >= best_params['depth_min']) &
        (df_filtered['keypoint_0_z'] <= best_params['depth_max'])
    ]
    
    # Apply volume filtering
    valid_indices = []
    for idx, row in df_filtered.iterrows():
        volume, _, _, _ = calculate_bbox_volume(row)
        if best_params['volume_min'] <= volume <= best_params['volume_max']:
            valid_indices.append(idx)
    
    df_optimal = df_filtered.loc[valid_indices]
    
    # Create visualization of instance distribution with optimal parameters
    frame_counts = {}
    for _, row in df_optimal.iterrows():
        video_frame_key = (row['video_filename'], row['frame_idx'])
        frame_counts[video_frame_key] = frame_counts.get(video_frame_key, 0) + 1
    
    counts = list(frame_counts.values())
    if len(counts) > 0:
        unique_counts, count_freq = np.unique(counts, return_counts=True)
        
        print(f"\n  Instance count distribution after optimization:")
        for count, freq in zip(unique_counts, count_freq):
            percentage = (freq / len(counts)) * 100
            marker = " 🎯" if count == 2 else ""
            print(f"    {count} instances: {freq} frames ({percentage:.1f}%){marker}")
    
    # Save optimal parameters to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    optimal_params_file = output_dir / "optimal_2instance_parameters.txt"
    with open(optimal_params_file, 'w') as f:
        f.write("OPTIMAL PARAMETERS FOR MAXIMUM 2-INSTANCE FRAMES\n")
        f.write("="*50 + "\n\n")
        f.write(f"Depth range: {best_params['depth_min']:.3f} to {best_params['depth_max']:.3f}\n")
        f.write(f"Volume range: {best_params['volume_min']:.3f} to {best_params['volume_max']:.3f}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Frames with 2 instances: {best_results['frames_with_2instances']}\n")
        f.write(f"  Total frames: {best_results['total_frames']}\n")
        f.write(f"  Percentage: {best_results['percentage']:.1f}%\n\n")
        f.write(f"Command to reproduce:\n")
        f.write(f"python src/preprocess/analyze_3d_poses.py \\\n")
        f.write(f"  --volume_min {best_params['volume_min']:.3f} \\\n")
        f.write(f"  --volume_max {best_params['volume_max']:.3f} \\\n")
        f.write(f"  --show_instances_only\n")
    
    print(f"\n💾 Optimal parameters saved to: {optimal_params_file}")
    
    return best_params, best_results


def analyze_3d_statistics(pose_3d_df):
    """Analyze 3D pose statistics"""
    print("\n=== 3D Pose Statistics ===")
    
    # Analyze all keypoint depths
    z_coords = []
    for i in range(17):
        z_coords.extend(pose_3d_df[f"keypoint_{i}_z"].values)
    z_coords = np.array(z_coords)
    
    print(f"\nAll keypoint depth (Z-axis) statistics:")
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
    parser = argparse.ArgumentParser(description="Analyze 3D pose predictions")
    parser.add_argument(
        "--depth_threshold",
        type=float,
        default=None,
        help="Depth threshold for filtering instances (only analyze instances with depth <= threshold)"
    )
    parser.add_argument(
        "--volume_min",
        type=float,
        default=None,
        help="Minimum volume threshold for filtering instances (only analyze instances with bbox volume >= threshold)"
    )
    parser.add_argument(
        "--volume_max",
        type=float,
        default=None,
        help="Maximum volume threshold for filtering instances (only analyze instances with bbox volume <= threshold)"
    )
    parser.add_argument(
        "--show_depth_only",
        action="store_true",
        help="Only show depth distribution analysis"
    )
    parser.add_argument(
        "--show_volume_only",
        action="store_true",
        help="Only show bounding box volume distribution analysis"
    )
    parser.add_argument(
        "--show_instances_only",
        action="store_true",
        help="Only show instances per frame analysis"
    )
    parser.add_argument(
        "--optimize_2instances",
        action="store_true",
        help="Optimize parameters to maximize frames with exactly 2 instances"
    )
    parser.add_argument(
        "--optimization_method",
        type=str,
        default="grid_search",
        choices=["grid_search", "scipy"],
        help="Optimization method: grid_search (thorough) or scipy (faster)"
    )
    
    args = parser.parse_args()
    
    # Load data
    pose_3d_df, pose_3d_norm_df = load_pose_data()
    
    if pose_3d_df is None:
        return
    
    print(f"Loaded {len(pose_3d_df)} 3D pose instances")
    
    # Run specific analysis if requested
    if args.show_depth_only:
        depth_df = analyze_depth_distribution(pose_3d_df, args.depth_threshold)
        print("\n=== Analysis Complete ===")
        return
    
    if args.show_volume_only:
        volume_df = analyze_bbox_volume_distribution(pose_3d_df, args.volume_min, args.volume_max)
        print("\n=== Analysis Complete ===")
        return
    
    if args.show_instances_only:
        instances_df = analyze_instances_per_frame(pose_3d_df, args.depth_threshold, args.volume_min, args.volume_max)
        print("\n=== Analysis Complete ===")
        return
    
    if args.optimize_2instances:
        optimal_params, optimal_results = run_optimization_analysis(pose_3d_df, method=args.optimization_method)
        print("\n=== Optimization Complete ===")
        return
    
    # Run full analysis pipeline
    # Always analyze depth distribution first
    depth_df = analyze_depth_distribution(pose_3d_df, args.depth_threshold)
    
    # Analyze bbox volume distribution
    volume_df = analyze_bbox_volume_distribution(pose_3d_df, args.volume_min, args.volume_max)
    
    # Analyze instances per frame (with optional depth and volume filtering)
    instances_df = analyze_instances_per_frame(pose_3d_df, args.depth_threshold, args.volume_min, args.volume_max)
    
    # Continue with full analysis
    # Analyze statistics
    # limb_lengths = analyze_3d_statistics(pose_3d_df)
    
    # Analyze pelvis-centered coordinate system
    # pelvis_distances = analyze_pelvis_centered_coordinates(pose_3d_df)
    
    # Analyze pose quality
    # analyze_3d_pose_quality(pose_3d_df)
    
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
    
    print("\n=== 3D Pose Analysis Complete ===")


if __name__ == "__main__":
    main()