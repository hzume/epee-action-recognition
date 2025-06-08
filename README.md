# Epee - Fencing Action Recognition System

フェンシング動画から特定のアクションを自動認識するシステム

## 認識可能なアクション

本システムは以下のフェンシングアクションを認識します：
- **Lunge** (突き)
- **Fleche** (フレッシュ)
- **Counter** (カウンター)
- **Parry** (パリー)
- **Prime** (プライム)

## ディレクトリ構造

```
epee/
│   configs/                    # 実験設定ファイル
│   ├── exp00_cnn.yaml         # CNN モデル設定
│   ├── exp03_lightgbm.yaml    # LightGBM 設定
│   └── exp04_lstm.yaml        # LSTM 設定
│
│   checkpoints/               # 学習済みモデル
│   ├── exp00/                 # 各実験のモデル
│   ├── exp01/
│   └── ...
│
│   input/                     # 入力データ
│   ├── metadata.json          # アノテーションID情報
│   ├── raw_annotation/        # 生のアノテーション
│   ├── videos/                # 動画ファイル
│   └── data_10hz/            # 前処理済みデータ
│       ├── frames/           # フレーム画像
│       └── pose_preds.csv    # ポーズ推定結果
│
│   output/                    # 出力結果
│   ├── predictions/          # 予測結果
│   └── visualizations/       # 可視化結果
│
│   scripts/                   # 実行スクリプト
│   ├── predict.py            # 単一動画予測CLI
│   └── predict_batch.py      # バッチ予測CLI
│
│   src/                       # ソースコード
│   ├── inference/            # 推論システム
│   │   ├── __init__.py
│   │   ├── api.py           # 統一API
│   │   ├── base.py          # 基底クラス
│   │   └── predictors/      # 各モデル実装
│   │       ├── cnn_predictor.py
│   │       ├── cnn_lstm_predictor.py
│   │       ├── lightgbm_predictor.py
│   │       └── lstm_predictor.py
│   │
│   ├── preprocess/           # 前処理モジュール
│   │   ├── video_to_frames.py      # 動画からフレーム抽出
│   │   ├── process_raw_annotation.py # アノテーション処理
│   │   ├── process_pose_preds.py    # ポーズ推定処理
│   │   └── run_preprocess.py       # 前処理実行スクリプト
│   │
│   ├── exp00/                # 実験00: CNN単体モデル
│   │   ├── dataset.py        # データセット定義
│   │   ├── model.py          # モデル定義
│   │   └── train.py          # 学習スクリプト
│   │
│   ├── exp01/                # 実験01: CNN-LSTM v1
│   │   └── ...               # 初期実装
│   │
│   ├── exp02/                # 実験02: CNN-LSTM v2
│   │   └── ...               # 改良版CNN-LSTM
│   │
│   ├── exp03/                # 実験03: LightGBM
│   │   ├── train.py          # 特徴量ベース学習
│   │   └── train_outcome.py  # アウトカム予測
│   │
│   ├── exp04/                # 実験04: LSTM 最終版
│   │   ├── train.py          # 時系列学習
│   │   └── train_outcome.py  # アウトカム予測
│   │
│   └── visualization/        # 可視化ツール
│       ├── show_pred_result.py      # 予測結果表示
│       └── show_pred_result_side.py # サイド情報表示
│
│   mmpose/                   # MMPose (ポーズ推定ライブラリ)
│   lightning_logs/           # PyTorch Lightning ログ
│   notebooks/                # Jupyter ノートブック
│   ├── eda.ipynb            # 探索的データ分析
│   └── pose_process_eda.ipynb # ポーズデータ分析
│
│   pyproject.toml           # プロジェクト設定
│   setup.sh                 # 環境構築スクリプト
│   pose_test.py             # ポーズ推定テスト
│   └── INFERENCE_README.md      # 推論システム詳細説明
```

## ファイル・ディレクトリの説明

### 設定関連
- **configs/**: 各実験の設定ファイル（YAML形式）。モデルパラメータや学習設定を管理
- **pyproject.toml**: Pythonプロジェクトの依存関係と環境設定

### データ関連
- **input/**: すべての入力データ
  - **raw_annotation/**: VIA (VGG Image Annotator) 形式のアノテーション
  - **videos/**: 元のフェンシング動画
  - **data_10hz/**: 10FPSに変換されたフレームとポーズデータ
- **output/**: モデルの予測結果と可視化

### 実験コード
- **exp00/**: ResNet34ベースのCNNによる単一フレーム分類
- **exp01-02/**: CNN-LSTMによる時系列考慮の初期実装
- **exp03/**: LightGBMによる特徴量ベースの高速推論
- **exp04/**: LSTMによる時系列学習の最終版実装

### 推論システム
- **src/inference/**: 統一的な推論フレームワーク
  - **base.py**: すべての予測器の基底クラス
  - **api.py**: 統一されたAPIインターフェース
  - **predictors/**: 各モデルの予測器実装

### 前処理
- **src/preprocess/**: データ前処理モジュール
  - 動画からフレーム抽出
  - アノテーション処理
  - ポーズ推定結果の処理

### ユーティリティ
- **scripts/**: 実行用スクリプト
- **pose_test.py**: MMPoseの動作確認とポーズ推定
- **notebooks/**: 分析と実験用ノートブック

## 環境構築

1. セットアップ
```bash
./setup.sh
```

2. ポーズ推定のテスト
```bash
python pose_test.py
```

3. モデル学習
```bash
python src/exp04/train.py
```

4. 推論実行
```bash
python scripts/predict.py input/videos/test.mp4 -e exp04 -o results.csv
```

詳細な推論システムの使い方は [INFERENCE_README.md](INFERENCE_README.md) を参照してください。

## 実験の流れ

1. **exp00**: 単純なCNN分類器の実装
2. **exp01-02**: 時系列を考慮したCNN-LSTM
3. **exp03**: ポーズ特徴量の統計的アプローチ
4. **exp04**: 時系列の全情報を活用した最終版

## ライセンス

[ライセンス情報をここに記載]