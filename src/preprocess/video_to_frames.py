from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


def save_frames_from_video(
    video_path: Path, output_dir: Path, extract_fps: int
):
    video_filename = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Error: '{video_path}' を開けませんでした。")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int(original_fps / extract_fps)
    frame_count = 0

    prog_bar = tqdm(
        total=total_frames, desc="Extracting frames...", leave=False
    )

    rows = {
        "frame_filename": [],
        "second": [],
    }
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = output_dir / "frames" / f"{video_filename}_{frame_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
            rows["frame_filename"].append(frame_path.name)
            rows["second"].append(frame_count / original_fps)

        frame_count += 1
        prog_bar.update(1)
    cap.release()
    prog_bar.close()

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    save_frames_from_video(
        Path("input/videos/2024-11-10-18-25-41.mp4"), Path("input/frames"), 30
    )
