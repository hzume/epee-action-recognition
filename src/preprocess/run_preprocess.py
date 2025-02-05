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


success_variations = ["success", "hit"]
failure_variations = ["failure", "miss", "fail"]


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
    frame_df["left_actions"] = ""
    frame_df["right_actions"] = ""
    frame_df["left_outcomes"] = ""
    frame_df["right_outcomes"] = ""


    for _, row in video_label_df[
        video_label_df["video_filename"] == video_filename
    ].iterrows():
        start_time, end_time = row["start_time"], row["end_time"]
        is_labeled = frame_df["second"].between(start_time, end_time)

        outcome = row["outcome"]
        if outcome in success_variations:
            outcome = "success"
        elif outcome in failure_variations:
            outcome = "failure"
        else:
            raise ValueError(f"Invalid outcome: {outcome}")
        
        if row["side"] == "r":
            frame_df.loc[is_labeled, "right_actions"] = frame_df.loc[is_labeled, "right_actions"].apply(lambda x: x + "," + row["action"] if x != "" else row["action"])
            frame_df.loc[is_labeled, "right_outcomes"] = frame_df.loc[is_labeled, "right_outcomes"].apply(lambda x: x + "," + outcome if x != "" else outcome)
        elif row["side"] == "l":
            frame_df.loc[is_labeled, "left_actions"] = frame_df.loc[is_labeled, "left_actions"].apply(lambda x: x + "," + row["action"] if x != "" else row["action"])
            frame_df.loc[is_labeled, "left_outcomes"] = frame_df.loc[is_labeled, "left_outcomes"].apply(lambda x: x + "," + outcome if x != "" else outcome)
        else:
            raise ValueError(f"Invalid side: {row['side']}")

    return frame_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", "-v", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--annotation_json_path", "-a", type=Path, required=True)
    parser.add_argument("--extract_fps", "-f", type=int, required=True)
    args = parser.parse_args()
    args = Args(**vars(args))

    # (args.output_dir / "frames").mkdir(parents=True, exist_ok=False)

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
