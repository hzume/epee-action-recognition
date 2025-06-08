# 3D Pose Data Processing with Optimization

## Complete Workflow

### 1. First, analyze and optimize parameters for maximum 2-instance frames:

```bash
# Run optimization to find best depth and volume parameters
python src/preprocess/analyze_3d_poses.py --optimize_2instances

# This will save optimal parameters to output/optimal_2instance_parameters.txt
```

### 2. Use optimal parameters to prepare training data:

```bash
# Example with optimized parameters (replace with actual values from step 1)
python src/preprocess/prepare_3d_data_for_training.py \
  --depth_min 2.150 \
  --depth_max 4.230 \
  --volume_min 0.420 \
  --volume_max 1.850 \
  --output_suffix "_2instances_optimized"
```

### 3. Alternative workflows:

```bash
# Run with custom parameters
python src/preprocess/prepare_3d_data_for_training.py \
  --depth_min 1.0 \
  --depth_max 5.0 \
  --volume_min 0.5 \
  --volume_max 2.0 \
  --output_suffix "_custom"

# Run without filtering (original behavior)
python src/preprocess/prepare_3d_data_for_training.py

# Quick analysis of current data
python src/preprocess/analyze_3d_poses.py --show_instances_only
```

## Output Structure

The filtered training data will be saved to:
- `input/data_10hz_3d_2instances_optimized/` (with optimal parameters)
- `input/data_10hz_3d_custom/` (with custom parameters)
- `input/data_10hz_3d/` (without filtering)

Each directory contains:
- `train_3d.csv` - Training data
- `val_3d.csv` - Validation data  
- `test_3d.csv` - Test data
- `scaler_3d.pkl` - Feature scaler
- `feature_info_3d.json` - Feature metadata
- `filtering_summary.txt` - Applied filtering summary

## Benefits

1. **Quality Data**: Only frames with exactly 2 instances (dual fencer scenes)
2. **Optimized Filtering**: Parameters chosen to maximize useful training data
3. **Reproducible**: All parameters and results saved for reference
4. **Flexible**: Can adjust parameters based on specific requirements