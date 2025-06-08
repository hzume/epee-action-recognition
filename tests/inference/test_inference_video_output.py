"""Integration tests for inference system with video output."""

import pytest
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from src.inference.base import BasePredictor
from src.inference.api import load_predictor


class MockPredictor(BasePredictor):
    """Mock predictor for testing."""
    
    def _load_model(self):
        """Return a mock model."""
        return Mock()
    
    def _preprocess_video(self, video_path: Path):
        """Mock video preprocessing."""
        return {
            'frames': np.random.rand(10, 3, 224, 224),  # 10 frames
            'video_path': video_path
        }
    
    def _predict_batch(self, batch: dict) -> np.ndarray:
        """Mock prediction."""
        num_frames = batch['frames'].shape[0]
        # Return random predictions for left and right players
        return np.random.randint(0, 6, size=(num_frames, 2))
    
    def _format_results(self, video_path: Path, predictions: np.ndarray, data: dict) -> pd.DataFrame:
        """Format mock predictions."""
        num_frames = predictions.shape[0]
        results = {
            'video_filename': [video_path.name] * num_frames,
            'frame_idx': list(range(num_frames)),
            'left_pred_action': [self.id_to_action[p[0]] for p in predictions],
            'right_pred_action': [self.id_to_action[p[1]] for p in predictions],
        }
        return pd.DataFrame(results)


class TestInferenceVideoOutput:
    """Test inference system with video output."""
    
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
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, (320, 240), 50, (255, 255, 255), -1)
            writer.write(frame)
        
        writer.release()
        return video_path
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock config file."""
        config_path = temp_dir / "config.yaml"
        config_content = """
inference:
  checkpoint: /fake/checkpoint.ckpt
  device: cpu
  
action_names:
  - lunge
  - fleche
  - counter
  - parry
  - prime
  - none
  
fps: 10
"""
        config_path.write_text(config_content)
        return config_path
    
    def test_predict_video_with_output(self, sample_video, mock_config, temp_dir):
        """Test predict_video with video output."""
        predictor = MockPredictor(mock_config)
        
        output_csv = temp_dir / "predictions.csv"
        output_video = temp_dir / "labeled_video.mp4"
        
        # Run prediction with video output
        results = predictor.predict_video(
            video_path=sample_video,
            output_path=output_csv,
            output_video_path=output_video
        )
        
        # Check CSV output
        assert output_csv.exists()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 10  # Mock predictor returns 10 frames
        assert all(col in results.columns for col in ['video_filename', 'frame_idx', 'left_pred_action', 'right_pred_action'])
        
        # Check video output
        assert output_video.exists()
        
        # Verify video properties
        cap = cv2.VideoCapture(str(output_video))
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 30  # Same as input
        cap.release()
    
    def test_predict_video_csv_only(self, sample_video, mock_config, temp_dir):
        """Test predict_video with only CSV output."""
        predictor = MockPredictor(mock_config)
        
        output_csv = temp_dir / "predictions.csv"
        
        results = predictor.predict_video(
            video_path=sample_video,
            output_path=output_csv
        )
        
        assert output_csv.exists()
        assert not (temp_dir / "labeled_video.mp4").exists()
    
    def test_predict_video_video_only(self, sample_video, mock_config, temp_dir):
        """Test predict_video with only video output."""
        predictor = MockPredictor(mock_config)
        
        output_video = temp_dir / "labeled_video.mp4"
        
        results = predictor.predict_video(
            video_path=sample_video,
            output_video_path=output_video
        )
        
        assert output_video.exists()
        assert isinstance(results, pd.DataFrame)
    
    def test_predict_batch_videos(self, temp_dir, mock_config):
        """Test batch video prediction."""
        # Create multiple videos
        video_paths = []
        for i in range(3):
            video_path = temp_dir / f"video_{i}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
            
            for j in range(15):  # Shorter videos for faster test
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                writer.write(frame)
            
            writer.release()
            video_paths.append(video_path)
        
        predictor = MockPredictor(mock_config)
        output_dir = temp_dir / "output"
        
        results = predictor.predict_batch_videos(video_paths, output_dir)
        
        # Check results
        assert len(results) == 3
        for i, video_path in enumerate(video_paths):
            video_name = video_path.stem
            assert video_name in results
            assert results[video_name] is not None
            
            # Check CSV files
            csv_path = output_dir / f"{video_name}_predictions.csv"
            assert csv_path.exists()
    
    @patch('src.inference.base.create_labeled_video')
    def test_video_output_error_handling(self, mock_create_video, sample_video, mock_config, temp_dir):
        """Test error handling in video output."""
        predictor = MockPredictor(mock_config)
        
        # Make create_labeled_video raise an error
        mock_create_video.side_effect = Exception("Video creation failed")
        
        output_csv = temp_dir / "predictions.csv"
        output_video = temp_dir / "labeled_video.mp4"
        
        # Should not raise error, but video won't be created
        with pytest.raises(Exception, match="Video creation failed"):
            results = predictor.predict_video(
                video_path=sample_video,
                output_path=output_csv,
                output_video_path=output_video
            )


@pytest.mark.parametrize("experiment", ["exp00", "exp03", "exp04"])
def test_load_predictor_interface(experiment, tmp_path):
    """Test that all predictors support video output interface."""
    # Create mock config
    config_path = tmp_path / f"{experiment}.yaml"
    config_content = f"""
experiment: {experiment}
inference:
  checkpoint: /fake/checkpoint.ckpt
  device: cpu

action_names:
  - lunge
  - fleche
  - counter
  - parry  
  - prime
  - none
"""
    config_path.write_text(config_content)
    
    # Mock the predictor classes to avoid loading real models
    with patch('src.inference.predictors.cnn_predictor.CNNPredictor', MockPredictor), \
         patch('src.inference.predictors.lightgbm_predictor.LightGBMPredictor', MockPredictor), \
         patch('src.inference.predictors.lstm_predictor.LSTMPredictor', MockPredictor):
        
        predictor = load_predictor(experiment, config_path=config_path)
        
        # Check that predict_video method accepts video output parameter
        import inspect
        sig = inspect.signature(predictor.predict_video)
        assert 'output_video_path' in sig.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])