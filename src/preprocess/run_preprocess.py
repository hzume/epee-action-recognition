import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from process_raw_annotation import make_video_label
from tqdm import tqdm
from video_to_frames import save_frames_from_video


@dataclass
class Args:
    video_dir: Path
    output_dir: Path
    annotation_json_path: Path
    extract_fps: int


def process_single_video(
    video_path: Path, output_dir: Path, extract_fps: int, video_label_df: pd.DataFrame
) -> pd.DataFrame:
    video_filename = video_path.name
    frame_df = save_frames_from_video(video_path, output_dir, extract_fps)
    frame_df["label"] = "None"
    frame_df["side"] = "None"
    frame_df["action"] = "None"
    frame_df["outcome"] = "None"

    for _, row in video_label_df[
        video_label_df["video_filename"] == video_filename
    ].iterrows():
        start_time, end_time = row["start_time"], row["end_time"]
        is_labeled = frame_df["second"].between(start_time, end_time)
        frame_df.loc[is_labeled, "label"] = row["label"]
        frame_df.loc[is_labeled, "side"] = row["side"]
        frame_df.loc[is_labeled, "action"] = row["action"]
        frame_df.loc[is_labeled, "outcome"] = row["outcome"]

    return frame_df


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


    process_video = partial(
        process_single_video,
        output_dir=args.output_dir,
        extract_fps=args.extract_fps,
        video_label_df=video_label_df
    )

    pool = Pool(processes=os.cpu_count())

    frame_dfs = list(tqdm(
            pool.imap(process_video, video_paths),
            total=len(video_paths),
            desc="Preprocessing videos"
        ))
    frame_label_df = pd.concat(frame_dfs)
    
    frame_label_df[["video_filename", "frame_idx"]] = frame_label_df["frame_filename"].str.extract(r"(.+)_(\d+)\.jpg")
    frame_label_df["video_filename"] = frame_label_df["video_filename"] + ".mp4"

    commit_hash = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    )

    params = {
        "commit_hash": commit_hash,
        "extract_fps": args.extract_fps,
        "video_files": [video_path.name for video_path in video_paths],
    }

    with open(args.output_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    frame_label_df.to_csv(args.output_dir / "frame_label.csv", index=False)
    video_label_df.to_csv(args.output_dir / "video_label.csv", index=False)


if __name__ == "__main__":
    main()
