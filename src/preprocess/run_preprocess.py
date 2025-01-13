import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from process_raw_label import make_video_label
from tqdm import tqdm
from video_to_frames import save_frames_from_video


@dataclass
class Args:
    video_dir: Path
    output_dir: Path
    annotation_json_path: Path
    extract_fps: int

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", "-v", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--annotation_json_path", "-a", type=Path, required=True)
    parser.add_argument("--extract_fps", "-f", type=int, required=True)
    args = parser.parse_args()
    args = Args(**vars(args))

    (args.output_dir / "frames").mkdir(parents=True, exist_ok=False)

    video_label_df = make_video_label(args.annotation_json_path)

    video_paths = list(args.video_dir.glob("*.mp4"))

    frame_dfs = []
    for video_path in tqdm(video_paths, desc="Preprocessing videos"):
        video_filename = video_path.name
        frame_df = save_frames_from_video(video_path, args.output_dir, args.extract_fps)
        frame_df["label"] = "None"
        frame_df["side"] = "None"
        frame_df["action"] = "None"
        frame_df["outcome"] = "None"
    
        for _, row in video_label_df[video_label_df["video_filename"] == video_filename].iterrows():
            start_time, end_time = row["start_time"], row["end_time"]
            frame_df.loc[frame_df["second"].between(start_time, end_time), "label"] = row["label"]
            frame_df.loc[frame_df["second"].between(start_time, end_time), "side"] = row["side"]
            frame_df.loc[frame_df["second"].between(start_time, end_time), "action"] = row["action"]
            frame_df.loc[frame_df["second"].between(start_time, end_time), "outcome"] = row["outcome"]

        frame_dfs.append(frame_df)
    frame_label_df = pd.concat(frame_dfs)

    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    params = {
        "commit_hash": commit_hash,
        "extract_fps": args.extract_fps,
        "video_files": [video_path.name for video_path in video_paths],
    }

    with open(args.output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    frame_label_df.to_csv(args.output_dir / "frame_label.csv", index=False)
    video_label_df.to_csv(args.output_dir / "video_label.csv", index=False)

if __name__=="__main__":
    main()