# 統一推論システム使用ガイド

このドキュメントでは、新しい統一推論システムの使い方を説明します。

## 概要

統一推論システムは、各実験（exp00-04）で訓練したモデルを簡単に切り替えて推論できるように設計されています。

## ディレクトリ構造

```
epee/
├── configs/              # 実験ごとの設定ファイル
│   ├── exp00_cnn.yaml
│   ├── exp01_cnn_lstm.yaml
│   ├── exp02_cnn_lstm.yaml
│   ├── exp03_lightgbm.yaml
│   └── exp04_lstm.yaml
├── checkpoints/          # 訓練済みモデル（要配置）
│   ├── exp00/
│   ├── exp01/
│   └── ...
├── scripts/              # 実行スクリプト
│   ├── predict.py        # 単一動画推論
│   └── predict_batch.py  # バッチ推論
└── src/inference/        # 推論システムの実装
```

## セットアップ

1. 訓練済みモデルをcheckpointsディレクトリに配置：
```bash
mkdir -p checkpoints/exp04
cp lightning_logs/version_XX/checkpoints/best.ckpt checkpoints/exp04/
```

2. 設定ファイルを必要に応じて編集

## 使い方

### 1. コマンドライン（CLI）での使用

#### 単一動画の推論
```bash
# デフォルト（exp04）での推論
python scripts/predict.py input/videos/test_video.mp4 -o results/predictions.csv

# 実験を指定して推論
python scripts/predict.py input/videos/test_video.mp4 -e exp03 -o results/predictions.csv

# ラベル付き動画も出力
python scripts/predict.py input/videos/test_video.mp4 \
    -o results/predictions.csv \
    --video-output results/labeled_video.mp4

# カスタム設定とチェックポイントを使用
python scripts/predict.py input/videos/test_video.mp4 \
    --config configs/custom.yaml \
    --checkpoint checkpoints/custom_model.ckpt \
    -o results/predictions.csv
```

#### バッチ推論
```bash
# ディレクトリ内の全動画を処理
python scripts/predict_batch.py \
    --video-dir input/videos/ \
    --output-dir results/ \
    --experiment exp04

# 特定の動画リストを処理
python scripts/predict_batch.py \
    --video-list video1.mp4 video2.mp4 video3.mp4 \
    --output-dir results/ \
    --experiment exp04

# 既存の結果をスキップ
python scripts/predict_batch.py \
    --video-dir input/videos/ \
    --output-dir results/ \
    --skip-existing

# ラベル付き動画も生成（predict_batch_with_video.py使用）
python scripts/predict_batch_with_video.py \
    --video-dir input/videos/ \
    --output-dir results/ \
    --video-output \
    --video-output-dir results/labeled_videos/
```

### 2. PythonスクリプトからのAPI使用

```python
from src.inference import load_predictor

# 推論器をロード
predictor = load_predictor("exp04")

# 単一動画の推論
results = predictor.predict_video("input/videos/test_video.mp4")
print(results.head())

# 結果をCSVに保存
results.to_csv("predictions.csv", index=False)

# ラベル付き動画も生成
results = predictor.predict_video(
    "input/videos/test_video.mp4",
    output_path="predictions.csv",
    output_video_path="labeled_video.mp4"
)

# バッチ推論
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
all_results = predictor.predict_batch_videos(video_paths, output_dir="results/")
```

### 3. ラベル付き動画の生成

```python
from src.inference.utils import create_labeled_video

# 予測結果からラベル付き動画を生成
create_labeled_video(
    video_path="input/videos/test_video.mp4",
    predictions_df=results,  # 予測結果のDataFrame
    output_path="output/labeled_video.mp4",
    fps=10,  # 予測のフレームレート
    font_scale=1.0,  # フォントサイズ
    left_color=(0, 0, 255),  # 左プレイヤーのラベル色（赤）
    right_color=(0, 255, 0)  # 右プレイヤーのラベル色（緑）
)
```

### 4. カスタム設定での使用

```python
# カスタム設定とチェックポイントを指定
predictor = load_predictor(
    "exp04",
    config_path="configs/custom_config.yaml",
    checkpoint_path="checkpoints/custom_model.ckpt"
)
```

## 各実験の特徴

- **exp00**: ResNet34d CNNベースライン（単一アクション分類）
- **exp01/02**: CNN-LSTM（左右プレイヤー別アクション分類）
- **exp03**: LightGBM（姿勢特徴量ベース）
- **exp04**: LSTM（骨格特徴量ベース）※最新

## 注意事項

1. **姿勢推定の事前実行**：exp03とexp04は姿勢推定結果（pose_preds.csv）が必要です
2. **モデルチェックポイント**：各実験のチェックポイントファイルを適切に配置してください
3. **設定ファイル**：必要に応じてconfigs/内の設定ファイルを編集してください

## トラブルシューティング

### チェックポイントが見つからない
```
Error: Checkpoint not found at checkpoints/exp04/best.ckpt
```
→ 訓練済みモデルを指定された場所に配置してください

### 姿勢データが見つからない
```
Error: Pose predictions not found: input/data_10hz/pose_preds.csv
```
→ pose_test.pyを実行して姿勢推定を行ってください

### CUDAメモリ不足
```
RuntimeError: CUDA out of memory
```
→ バッチサイズを小さくするか、--device cpuオプションを使用してください

## 今後の拡張予定

- [x] 可視化機能の実装（ラベル付き動画出力）
- [ ] リアルタイム推論のサポート
- [ ] Webインターフェースの追加
- [ ] 残りの予測器（CNN, CNN-LSTM, LightGBM）の実装