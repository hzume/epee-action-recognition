from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_result_path", type=str, default="pred_result.csv")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_result_path)
    data_dir = Path("input/data_10hz")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_filenames = df["video_filename"].unique()
    for video_filename in video_filenames:
        actions = df[df["video_filename"] == video_filename]["action"].values
        pred_actions = df[df["video_filename"] == video_filename]["pred_action"].values

        video_path = data_dir.parent / "videos" / video_filename
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"動画を開けませんでした: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_interval = int(fps * 0.1)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_dir / video_filename, fourcc, fps, (width, height))

        # フレームカウンタ & ラベルインデックス
        frame_count = 0
        label_index = 0

        while True:
            # 1フレーム読み込み
            ret, frame = cap.read()
            if not ret:
                # 動画終了、または読み込みエラー
                break

            # もし frame_count が frame_interval の倍数であれば、ラベルを更新
            # 例: frame_interval=3 なら frame_count=0,3,6,9,... のタイミング
            if frame_count % frame_interval == 0:
                # labels にまだラベルが残っている場合のみ更新
                if label_index < len(actions):
                    current_true_action = actions[label_index]
                    current_pred_action = pred_actions[label_index]
                    label_index += 1
                else:
                    # ラベルが足りない場合は最後のラベルを使い続ける、あるいは空文字等も可
                    current_true_action = actions[-1] if len(actions) > 0 else ""
                    current_pred_action = pred_actions[-1] if len(pred_actions) > 0 else ""
            # ラベルをフレームに描画
            cv2.putText(
                img=frame,
                text=str(current_true_action),
                org=(50, 50),  # テキスト表示位置 (x, y)
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255),  # (B, G, R) で赤
                thickness=2,
                lineType=cv2.LINE_AA
            )
            
            cv2.putText(
                img=frame,
                text=str(current_pred_action),
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
