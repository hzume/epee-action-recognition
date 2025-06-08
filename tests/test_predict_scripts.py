"""Tests for command-line prediction scripts."""

import pytest
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import cv2
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock


class TestPredictScripts:
    """Test command-line prediction scripts."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_video(self, temp_dir):
        """Create a sample video."""
        video_path = temp_dir / "test_video.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        for i in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
            writer.write(frame)
        
        writer.release()
        return video_path
    
    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor."""
        predictor = Mock()
        
        # Mock predict_video method
        def mock_predict_video(video_path, output_path=None, output_video_path=None):
            results = pd.DataFrame({
                'video_filename': [Path(video_path).name] * 3,
                'frame_idx': [0, 1, 2],
                'left_pred_action': ['lunge', 'parry', 'none'],
                'right_pred_action': ['counter', 'none', 'fleche']
            })
            
            if output_path:
                results.to_csv(output_path, index=False)
            
            if output_video_path:
                # Create a dummy video file
                Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
                Path(output_video_path).touch()
            
            return results
        
        predictor.predict_video = mock_predict_video
        predictor.config = {'fps': 10}
        return predictor
    
    @patch('src.inference.api.load_predictor')
    def test_predict_single_video_csv_output(self, mock_load_predictor, mock_predictor, sample_video, temp_dir):
        """Test predict.py with CSV output."""
        mock_load_predictor.return_value = mock_predictor
        
        output_csv = temp_dir / "predictions.csv"
        
        # Run script
        result = subprocess.run([
            sys.executable, "scripts/predict.py",
            str(sample_video),
            "-o", str(output_csv),
            "-e", "exp04"
        ], capture_output=True, text=True)
        
        # Check result
        assert result.returncode == 0
        assert "Predictions saved to:" in result.stderr
        assert output_csv.exists()
    
    @patch('src.inference.api.load_predictor')
    def test_predict_single_video_with_video_output(self, mock_load_predictor, mock_predictor, sample_video, temp_dir):
        """Test predict.py with video output."""
        mock_load_predictor.return_value = mock_predictor
        
        output_csv = temp_dir / "predictions.csv"
        output_video = temp_dir / "labeled.mp4"
        
        # Run script
        result = subprocess.run([
            sys.executable, "scripts/predict.py", 
            str(sample_video),
            "-o", str(output_csv),
            "--video-output", str(output_video),
            "-e", "exp04"
        ], capture_output=True, text=True)
        
        # Check result
        assert result.returncode == 0
        assert "Predictions saved to:" in result.stderr
        assert "Labeled video saved to:" in result.stderr
        assert output_csv.exists()
        assert output_video.exists()
    
    @patch('src.inference.api.load_predictor')
    def test_predict_single_video_stdout(self, mock_load_predictor, mock_predictor, sample_video):
        """Test predict.py with stdout output."""
        mock_load_predictor.return_value = mock_predictor
        
        # Run script
        result = subprocess.run([
            sys.executable, "scripts/predict.py",
            str(sample_video),
            "-e", "exp04"
        ], capture_output=True, text=True)
        
        # Check result
        assert result.returncode == 0
        assert "video_filename,frame_idx,left_pred_action,right_pred_action" in result.stdout
        assert "test_video.mp4" in result.stdout
    
    def test_predict_nonexistent_video(self, temp_dir):
        """Test error handling for nonexistent video."""
        result = subprocess.run([
            sys.executable, "scripts/predict.py",
            str(temp_dir / "nonexistent.mp4"),
            "-e", "exp04"
        ], capture_output=True, text=True)
        
        assert result.returncode == 1
        assert "Video file not found" in result.stderr
    
    @patch('src.inference.api.load_predictor')
    def test_predict_batch_with_video_output(self, mock_load_predictor, temp_dir):
        """Test batch prediction with video output."""
        # Create mock predictor
        predictor = Mock()
        
        def mock_predict_batch(video_paths, output_dir):
            results = {}
            for video_path in video_paths:
                video_name = Path(video_path).stem
                df = pd.DataFrame({
                    'video_filename': [Path(video_path).name] * 2,
                    'frame_idx': [0, 1],
                    'left_pred_action': ['lunge', 'parry'],
                    'right_pred_action': ['counter', 'none']
                })
                
                # Save CSV
                csv_path = Path(output_dir) / f"{video_name}_predictions.csv"
                df.to_csv(csv_path, index=False)
                results[video_name] = df
            
            return results
        
        predictor.predict_batch_videos = mock_predict_batch
        predictor.config = {'fps': 10}
        mock_load_predictor.return_value = predictor
        
        # Create test videos
        video_dir = temp_dir / "videos"
        video_dir.mkdir()
        
        for i in range(2):
            video_path = video_dir / f"video_{i}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
            for _ in range(15):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
        
        output_dir = temp_dir / "output"
        
        # Mock create_labeled_video_batch to avoid actual video processing
        with patch('src.inference.utils.create_labeled_video_batch') as mock_create_batch:
            # Run script
            result = subprocess.run([
                sys.executable, "scripts/predict_batch_with_video.py",
                "--video-dir", str(video_dir),
                "-o", str(output_dir),
                "--video-output",
                "-e", "exp04"
            ], capture_output=True, text=True)
        
        # Check result
        assert result.returncode == 0
        assert "Found 2 video(s) to process" in result.stderr
        assert "Generating labeled videos..." in result.stderr
        
        # Check CSV outputs
        assert (output_dir / "video_0_predictions.csv").exists()
        assert (output_dir / "video_1_predictions.csv").exists()
        assert (output_dir / "all_predictions.csv").exists()


class TestScriptHelp:
    """Test script help messages."""
    
    def test_predict_help(self):
        """Test predict.py help."""
        result = subprocess.run([
            sys.executable, "scripts/predict.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Predict fencing actions from video" in result.stdout
        assert "--video-output" in result.stdout
        assert "--experiment" in result.stdout
    
    def test_predict_batch_help(self):
        """Test predict_batch.py help."""
        result = subprocess.run([
            sys.executable, "scripts/predict_batch.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Predict fencing actions from multiple videos" in result.stdout
        assert "--video-dir" in result.stdout
        assert "--video-list" in result.stdout
    
    def test_predict_batch_with_video_help(self):
        """Test predict_batch_with_video.py help."""
        result = subprocess.run([
            sys.executable, "scripts/predict_batch_with_video.py", "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "--video-output" in result.stdout
        assert "--video-output-dir" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])