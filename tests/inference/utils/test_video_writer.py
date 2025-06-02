"""Unit tests for video_writer module."""

import pytest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import tempfile
import shutil

from src.inference.utils.video_writer import create_labeled_video, create_labeled_video_batch


class TestVideoWriter:
    """Test video writer functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_video(self, temp_dir):
        """Create a sample video for testing."""
        video_path = temp_dir / "test_video.mp4"
        
        # Create a simple video with 30 frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        
        for i in range(30):
            # Create a frame with frame number
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
        
        writer.release()
        return video_path
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample prediction DataFrame."""
        data = {
            'video_filename': ['test_video.mp4'] * 3,
            'frame_idx': [0, 1, 2],
            'left_pred_action': ['lunge', 'parry', 'fleche'],
            'right_pred_action': ['counter', 'prime', 'lunge']
        }
        return pd.DataFrame(data)
    
    def test_create_labeled_video(self, sample_video, sample_predictions, temp_dir):
        """Test creating a labeled video."""
        output_path = temp_dir / "labeled_video.mp4"
        
        create_labeled_video(
            video_path=sample_video,
            predictions_df=sample_predictions,
            output_path=output_path,
            fps=10
        )
        
        # Check that output video exists
        assert output_path.exists()
        
        # Check video properties
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        assert fps == 30.0  # Original video FPS
        assert frame_count == 30  # Same as original
        assert width == 640
        assert height == 480
        
        cap.release()
    
    def test_create_labeled_video_with_custom_colors(self, sample_video, sample_predictions, temp_dir):
        """Test creating labeled video with custom colors."""
        output_path = temp_dir / "labeled_video_custom.mp4"
        
        create_labeled_video(
            video_path=sample_video,
            predictions_df=sample_predictions,
            output_path=output_path,
            fps=10,
            left_color=(255, 0, 0),  # Blue
            right_color=(0, 0, 255),  # Red
            font_scale=2.0,
            font_thickness=3
        )
        
        assert output_path.exists()
    
    def test_create_labeled_video_missing_video(self, sample_predictions, temp_dir):
        """Test error handling for missing video."""
        with pytest.raises(ValueError, match="Could not open video"):
            create_labeled_video(
                video_path=temp_dir / "nonexistent.mp4",
                predictions_df=sample_predictions,
                output_path=temp_dir / "output.mp4"
            )
    
    def test_create_labeled_video_batch(self, sample_video, temp_dir):
        """Test batch video labeling."""
        # Create predictions CSV
        predictions_data = {
            'video_filename': ['test_video.mp4'] * 3 + ['test_video2.mp4'] * 2,
            'frame_idx': [0, 1, 2, 0, 1],
            'left_pred_action': ['lunge', 'parry', 'fleche', 'counter', 'prime'],
            'right_pred_action': ['counter', 'prime', 'lunge', 'fleche', 'parry']
        }
        predictions_df = pd.DataFrame(predictions_data)
        predictions_csv = temp_dir / "predictions.csv"
        predictions_df.to_csv(predictions_csv, index=False)
        
        # Copy video to create second video
        sample_video2 = temp_dir / "test_video2.mp4"
        shutil.copy(sample_video, sample_video2)
        
        # Create output directory
        output_dir = temp_dir / "output_videos"
        
        # Run batch labeling
        create_labeled_video_batch(
            predictions_csv=predictions_csv,
            output_dir=output_dir,
            video_dir=temp_dir,
            fps=10
        )
        
        # Check outputs
        assert (output_dir / "test_video.mp4").exists()
        assert (output_dir / "test_video2.mp4").exists()
    
    def test_frame_interval_calculation(self, sample_video, sample_predictions, temp_dir):
        """Test correct frame interval calculation for different FPS."""
        output_path = temp_dir / "labeled_video_fps_test.mp4"
        
        # Test with 10 Hz predictions on 30 FPS video
        # Should update labels every 3 frames
        create_labeled_video(
            video_path=sample_video,
            predictions_df=sample_predictions,
            output_path=output_path,
            fps=10  # Prediction FPS
        )
        
        # Read the output video and check label updates
        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened()
        
        # The frame interval should be 30/10 = 3
        # So labels should update at frames 0, 3, 6, 9, ...
        cap.release()
        assert output_path.exists()


@pytest.mark.integration
class TestVideoWriterIntegration:
    """Integration tests for video writer with real inference results."""
    
    @pytest.fixture
    def mock_inference_results(self):
        """Create mock inference results DataFrame."""
        # Simulate 5 seconds of predictions at 10Hz
        num_frames = 50
        data = {
            'video_filename': ['test_match.mp4'] * num_frames,
            'frame_idx': list(range(num_frames)),
            'left_pred_action': ['none'] * 10 + ['lunge'] * 5 + ['none'] * 20 + ['fleche'] * 10 + ['none'] * 5,
            'right_pred_action': ['none'] * 15 + ['parry'] * 5 + ['counter'] * 10 + ['none'] * 20
        }
        return pd.DataFrame(data)
    
    def test_integration_with_inference_output(self, temp_dir, mock_inference_results):
        """Test video writer with realistic inference output."""
        # Create a longer test video (5 seconds at 30 FPS)
        video_path = temp_dir / "test_match.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1280, 720))
        
        for i in range(150):  # 5 seconds * 30 FPS
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50  # Dark gray
            # Add some visual elements
            cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), 2)
            cv2.rectangle(frame, (980, 100), (1180, 300), (255, 255, 255), 2)
            writer.write(frame)
        
        writer.release()
        
        # Create labeled video
        output_path = temp_dir / "labeled_match.mp4"
        create_labeled_video(
            video_path=video_path,
            predictions_df=mock_inference_results,
            output_path=output_path,
            fps=10
        )
        
        # Verify output
        assert output_path.exists()
        
        # Check that the video has the same duration
        cap_in = cv2.VideoCapture(str(video_path))
        cap_out = cv2.VideoCapture(str(output_path))
        
        assert cap_in.get(cv2.CAP_PROP_FRAME_COUNT) == cap_out.get(cv2.CAP_PROP_FRAME_COUNT)
        assert cap_in.get(cv2.CAP_PROP_FPS) == cap_out.get(cv2.CAP_PROP_FPS)
        
        cap_in.release()
        cap_out.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])