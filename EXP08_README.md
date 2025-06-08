# Exp08: 3D Pose-based Action Recognition

## 概要

Exp08は、3D姿勢推定結果を用いたフェンシング動作認識モデルです。2Dの画像座標ではなく、3D空間での関節位置情報を活用することで、より正確な動作理解を目指します。

## 主な特徴

### 1. 3D姿勢データの活用
- MMPoseのHuman3Dモデルで生成された3D姿勢推定結果を使用
- 17個のHuman3D関節点のx, y, z座標を利用（骨盤中心のローカル座標系）
- 深度情報による空間的関係の理解

### 2. 空間注意機構（Spatial Attention）
- 関節間の接続関係を考慮した注意機構
- Human3Dスケルトンの解剖学的構造を反映
- マルチヘッド注意による関節間相互作用のモデリング

### 3. 関節埋め込み（Joint Embedding）
- 各関節に特化した特徴埋め込み
- 関節ごとの特性を考慮した表現学習
- 身体部位の機能的差異を反映

### 4. 時空間モデリング
- LSTM による時系列モデリング
- 3D空間での動作の連続性を捉える
- 双方向LSTMによる前後文脈の活用

### 5. 3D特有の特徴量（Human3D対応）
- 3D関節角度の計算（肘、膝、肩、腰、背骨、首）
- 3D速度ベクトル
- プレイヤー間の3D距離
- 身体方向（torso、head、shoulder orientation）の推定
- 骨盤中心ローカル座標系の活用

### 6. Human3Dキーポイント構造
```
ID  キーポイント名      対応する人体部位
0   pelvis           骨盤中心（原点）
1   left_hip         左股関節
2   left_knee        左膝
3   left_ankle       左足首
4   right_hip        右股関節
5   right_knee       右膝
6   right_ankle      右足首
7   spine            背骨（腰椎）
8   thorax           胸部（胸椎）
9   neck             首
10  head             頭部中心
11  left_shoulder    左肩
12  left_elbow       左肘
13  left_wrist       左手首
14  right_shoulder   右肩
15  right_elbow      右肘
16  right_wrist      右手首
```

## アーキテクチャ

```
3D Pose Sequence (B, T, 17, 3)
          ↓
    Joint Embedding
          ↓
   Spatial Attention
          ↓
    Feature Fusion
          ↓
      Bi-LSTM
          ↓
  Temporal Attention
          ↓
   Classification Heads
    (Left & Right)
```

## データフロー

### 1. データ前処理
```bash
# 3D姿勢データの処理
python src/preprocess/process_pose_preds_3d.py

# 訓練用データの準備
python src/preprocess/prepare_3d_data_for_training.py
```

### 2. モデル訓練
```bash
# 全fold交差検証
python -m src.exp08.train --config configs/exp08_3d_pose_lstm.yaml

# 特定fold
python -m src.exp08.train --config configs/exp08_3d_pose_lstm.yaml --fold 0

# データ準備込み
python -m src.exp08.train --config configs/exp08_3d_pose_lstm.yaml --prepare_data
```

## 設定パラメータ

### モデル設定
- `hidden_size`: LSTM隠れ層サイズ (default: 512)
- `num_layers`: LSTM層数 (default: 3)
- `use_spatial_attention`: 空間注意機構の使用 (default: true)
- `use_joint_embeddings`: 関節埋め込みの使用 (default: true)
- `spatial_attention_heads`: 注意ヘッド数 (default: 8)

### データ処理
- `window_width`: 時間窓幅 (default: 5)
- `normalize_poses`: 姿勢正規化 (default: true)
- `center_on_hip`: 腰中心での正規化 (default: true)

### データ拡張
- `gaussian_noise_std`: ガウシアンノイズ (default: 0.01)
- `rotation_range`: 3D回転角度範囲 (default: 10度)
- `scale_range`: スケール変動範囲 (default: 0.1)

## 期待される改善点

### 1. 深度情報の活用
- 2Dでは捉えられない前後の動きを認識
- 奥行き方向の動作パターンの理解

### 2. 3D空間での関節関係
- より正確な関節角度の計算
- 3D空間での身体姿勢の理解

### 3. 視点不変性
- カメラ角度に依存しない特徴抽出
- より汎用的な動作認識

### 4. 精密な動作解析
- フェンシング特有の3D動作の捉獲
- 突きや受けの深度変化の検出

## 出力ファイル

### 訓練結果
- `outputs/exp08/fold_X/`: 各foldのモデルとログ
- `outputs/exp08/3d_cv_results.csv`: 交差検証結果
- `outputs/exp08/3d_predictions_fold_X.csv`: 各foldの予測結果

### 評価指標
- 全体精度と動作クラス別精度
- F1スコア（マクロ・重み付き）
- 3D特有の評価指標

## 前提条件

### データ要件
- 3D姿勢推定結果 (`output/pose_10hz_3d/`)
- フレームラベル (`input/data_10hz/frame_label.csv`)
- 処理済み3D訓練データ (`input/data_10hz_3d/`)

### 依存関係
- PyTorch Lightning
- scikit-learn
- scipy (3D処理用)

## 実行例

```bash
# 完全なワークフロー
python src/preprocess/process_pose_preds_3d.py
python src/preprocess/prepare_3d_data_for_training.py
python -m src.exp08.train --config configs/exp08_3d_pose_lstm.yaml

# デバッグモード（高速テスト）
python -m src.exp08.train --config configs/exp08_3d_pose_lstm.yaml --debug
```

## 比較実験

Exp08の結果をExp07（2D+temporal）と比較することで、3D情報の有効性を検証できます：

- **精度改善**: 3D情報による認識精度の向上
- **安定性**: 視点変化に対する頑健性
- **解釈性**: 3D空間での動作理解の向上