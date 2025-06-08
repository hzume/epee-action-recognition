#!/usr/bin/env python3
"""Run tests for video output functionality."""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests related to video output."""
    
    print("=" * 60)
    print("Running Video Output Tests")
    print("=" * 60)
    
    # Test categories
    test_suites = [
        {
            "name": "Unit Tests - Video Writer",
            "command": [sys.executable, "-m", "pytest", "tests/inference/utils/test_video_writer.py", "-v"]
        },
        {
            "name": "Integration Tests - Inference with Video Output", 
            "command": [sys.executable, "-m", "pytest", "tests/inference/test_inference_video_output.py", "-v"]
        },
        {
            "name": "Script Tests - Command Line Interface",
            "command": [sys.executable, "-m", "pytest", "tests/test_predict_scripts.py", "-v"]
        }
    ]
    
    all_passed = True
    
    for suite in test_suites:
        print(f"\n{'-' * 60}")
        print(f"Running: {suite['name']}")
        print(f"{'-' * 60}")
        
        result = subprocess.run(suite['command'], capture_output=False)
        
        if result.returncode != 0:
            all_passed = False
            print(f"\n❌ {suite['name']} FAILED")
        else:
            print(f"\n✅ {suite['name']} PASSED")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("=" * 60)
    
    return 0 if all_passed else 1


def run_quick_check():
    """Run a quick functionality check without full tests."""
    
    print("=" * 60)
    print("Quick Functionality Check")
    print("=" * 60)
    
    # Check imports
    print("\n1. Checking imports...")
    try:
        from src.inference.utils import create_labeled_video
        from src.inference import load_predictor
        print("   ✅ Imports successful")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return 1
    
    # Check video writer module
    print("\n2. Checking video writer module...")
    video_writer_path = Path("src/inference/utils/video_writer.py")
    if video_writer_path.exists():
        print(f"   ✅ {video_writer_path} exists")
    else:
        print(f"   ❌ {video_writer_path} not found")
        return 1
    
    # Check updated scripts
    print("\n3. Checking updated scripts...")
    scripts = [
        "scripts/predict.py",
        "scripts/predict_batch_with_video.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            # Check for video output option
            content = script_path.read_text()
            if "video-output" in content or "video_output" in content:
                print(f"   ✅ {script} has video output support")
            else:
                print(f"   ⚠️  {script} may not have video output support")
        else:
            print(f"   ❌ {script} not found")
    
    # Check base predictor update
    print("\n4. Checking base predictor...")
    base_path = Path("src/inference/base.py")
    if base_path.exists():
        content = base_path.read_text()
        if "output_video_path" in content:
            print("   ✅ Base predictor has video output support")
        else:
            print("   ❌ Base predictor missing video output support")
            return 1
    
    print("\n" + "=" * 60)
    print("✅ Quick check completed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run video output tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick functionality check instead of full tests")
    args = parser.parse_args()
    
    if args.quick:
        sys.exit(run_quick_check())
    else:
        sys.exit(run_tests())