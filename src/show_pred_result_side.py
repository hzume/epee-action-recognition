from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

if __name__ == "__main__":
    executed_dir = Path(__file__).parent
    parser = ArgumentParser()
    parser.add_argument("--pred_result_path", type=str, default="pred_result.csv")
    parser.add_argument("--output_dir", type=str, default=executed_dir / "output")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    pred_fps = args.fps

    df = pd.read_csv(args.pred_result_path)
    data_dir = Path("input/data_10hz")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_filenames = df["video_filename"].unique()
    for video_filename in video_filenames:
        frame_idx = df[df["video_filename"] == video_filename]["frame_idx"].values
        left_pred_actions = df[df["video_filename"] == video_filename]["left_pred_action"].values
        right_pred_actions = df[df["video_filename"] == video_filename]["right_pred_action"].values

        video_path = data_dir.parent / "videos" / video_filename
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"動画を開けませんでした: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_interval = int(fps / pred_fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_dir / video_filename, fourcc, fps, (width, height))

        # フレームカウンタ & ラベルインデックス
        frame_count = 0
        pred_frame_count = 0
        left_label_index = 0
        right_label_index = 0

        while True:
            # 1フレーム読み込み
            ret, frame = cap.read()
            if not ret:
                # 動画終了、または読み込みエラー
                break

            # もし frame_count が frame_interval の倍数であれば、ラベルを更新
            # 例: frame_interval=3 なら frame_count=0,3,6,9,... のタイミング
            if frame_count % frame_interval == 0:
                if pred_frame_count in frame_idx:
                    current_left_pred_action = left_pred_actions[frame_idx.tolist().index(pred_frame_count)]
                else:
                    current_left_pred_action = ""
                if pred_frame_count in frame_idx:
                    current_right_pred_action = right_pred_actions[frame_idx.tolist().index(pred_frame_count)]
                else:
                    current_right_pred_action = ""
                pred_frame_count += 1
            # ラベルをフレームに描画
            cv2.putText(
                img=frame,
                text="left: " + str(current_left_pred_action),
                org=(50, 50),  # テキスト表示位置 (x, y)
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255),  # (B, G, R) で赤
                thickness=2,
                lineType=cv2.LINE_AA
            )
            
            cv2.putText(
                img=frame,
                text="right: " + str(current_right_pred_action),
                org=(50, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # フレームを書き込み
            writer.write(frame)
            frame_count += 1

        # 後処理
        cap.release()
        writer.release()
        print(f"出力動画ファイル: {output_dir / video_filename}")
