import json
import pandas as pd
from pathlib import Path

def make_video_label(file_path: Path) -> pd.DataFrame:
    with open(file_path, "r") as f:
        data = json.load(f)
    vid_list = data["project"]["vid_list"]
    file = data["file"]
    vid2vname: dict[str, str] = {vid: file[vid]["fname"] for vid in vid_list}

    metadata = data["metadata"]
    rows: dict[str, list[str | float]] = {
        "id": [],
        "video_filename": [],
        "start_time": [],
        "end_time": [],
        "label": [],
        "side": [],
        "action": [],
        "outcome": [],
    }
    for scene_id, meta in metadata.items():
        for action_id, label in meta["av"].items():
            rows["id"].append(f"{scene_id}_{action_id}")
            rows["video_filename"].append(vid2vname[meta["vid"]])
            rows["start_time"].append(meta["z"][0])
            rows["end_time"].append(meta["z"][1])
            rows["label"].append(label)
            side, action, outcome = label.split("_")
            rows["side"].append(side)
            rows["action"].append(action)
            rows["outcome"].append(outcome)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = make_video_label(Path("input/raw_labels/via_project_13Jan2025_01h17m28s.json"))
    print(df)