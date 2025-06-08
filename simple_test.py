#!/usr/bin/env python3
"""Simple test without pytest - checks basic functionality."""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.inference.utils import create_labeled_video
        print("  ✅ Imported create_labeled_video")
    except Exception as e:
        print(f"  ❌ Failed to import create_labeled_video: {e}")
        return False
    
    try:
        from src.inference import load_predictor
        print("  ✅ Imported load_predictor")
    except Exception as e:
        print(f"  ❌ Failed to import load_predictor: {e}")
        return False
    
    try:
        from src.inference.base import BasePredictor
        print("  ✅ Imported BasePredictor")
    except Exception as e:
        print(f"  ❌ Failed to import BasePredictor: {e}")
        return False
    
    return True


def test_video_writer_function():
    """Test video writer function signature."""
    print("\nTesting video writer function...")
    
    try:
        from src.inference.utils.video_writer import create_labeled_video
        import inspect
        
        sig = inspect.signature(create_labeled_video)
        params = list(sig.parameters.keys())
        
        required_params = ['video_path', 'predictions_df', 'output_path']
        for param in required_params:
            if param in params:
                print(f"  ✅ Has parameter: {param}")
            else:
                print(f"  ❌ Missing parameter: {param}")
                return False
        
        return True
    except Exception as e:
        print(f"  ❌ Error checking function: {e}")
        return False


def test_base_predictor_update():
    """Test that BasePredictor has video output support."""
    print("\nTesting BasePredictor update...")
    
    try:
        from src.inference.base import BasePredictor
        import inspect
        
        sig = inspect.signature(BasePredictor.predict_video)
        params = list(sig.parameters.keys())
        
        if 'output_video_path' in params:
            print("  ✅ BasePredictor.predict_video has output_video_path parameter")
            return True
        else:
            print("  ❌ BasePredictor.predict_video missing output_video_path parameter")
            return False
    except Exception as e:
        print(f"  ❌ Error checking BasePredictor: {e}")
        return False


def test_mock_video_creation():
    """Test creating a mock video with labels."""
    print("\nTesting mock video creation...")
    
    temp_dir = None
    try:
        # Import OpenCV
        try:
            import cv2
        except ImportError:
            print("  ⚠️  OpenCV not installed, skipping video creation test")
            print("     Install with: pip install opencv-python")
            return True
        
        from src.inference.utils.video_writer import create_labeled_video
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create a simple video
        video_path = temp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        # Write 30 frames (1 second at 30 FPS)
        for i in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
            writer.write(frame)
        
        writer.release()
        print("  ✅ Created test video")
        
        # Create mock predictions
        predictions = pd.DataFrame({
            'video_filename': ['test_video.mp4'] * 3,
            'frame_idx': [0, 1, 2],
            'left_pred_action': ['lunge', 'parry', 'fleche'],
            'right_pred_action': ['counter', 'prime', 'none']
        })
        print("  ✅ Created mock predictions")
        
        # Create labeled video
        output_path = temp_path / "labeled_video.mp4"
        create_labeled_video(
            video_path=video_path,
            predictions_df=predictions,
            output_path=output_path,
            fps=10
        )
        
        if output_path.exists():
            print("  ✅ Created labeled video")
            
            # Check video properties
            cap = cv2.VideoCapture(str(output_path))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"  ✅ Output video has {frame_count} frames")
                cap.release()
            else:
                print("  ❌ Could not open output video")
                return False
        else:
            print("  ❌ Output video not created")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error during video creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)


def test_cli_scripts():
    """Test that CLI scripts have video output options."""
    print("\nTesting CLI scripts...")
    
    scripts = {
        'scripts/predict.py': '--video-output',
        'scripts/predict_batch_with_video.py': '--video-output'
    }
    
    all_ok = True
    for script_path, expected_arg in scripts.items():
        path = Path(script_path)
        if path.exists():
            content = path.read_text()
            if expected_arg in content:
                print(f"  ✅ {script_path} has {expected_arg} option")
            else:
                print(f"  ❌ {script_path} missing {expected_arg} option")
                all_ok = False
        else:
            print(f"  ❌ {script_path} not found")
            all_ok = False
    
    return all_ok


def main():
    """Run all simple tests."""
    print("=" * 60)
    print("Running Simple Functionality Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Video Writer Function", test_video_writer_function),
        ("Base Predictor Update", test_base_predictor_update),
        ("Mock Video Creation", test_mock_video_creation),
        ("CLI Scripts", test_cli_scripts),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())