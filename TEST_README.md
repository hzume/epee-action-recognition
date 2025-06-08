# テストガイド - 動画出力機能

このドキュメントでは、推論システムの動画出力機能のテスト方法を説明します。

## テストの概要

実装した動画出力機能に対して、以下の3種類のテストを作成しました：

1. **ユニットテスト** (`tests/inference/utils/test_video_writer.py`)
   - `create_labeled_video`関数の基本動作
   - エラーハンドリング
   - バッチ処理機能

2. **統合テスト** (`tests/inference/test_inference_video_output.py`)
   - 推論システムとの統合
   - CSV出力と動画出力の組み合わせ
   - 各実験モデルの互換性

3. **CLIテスト** (`tests/test_predict_scripts.py`)
   - コマンドラインスクリプトの動作
   - 引数の処理
   - エラーメッセージ

## テストの実行方法

### 前提条件

```bash
# 必要なパッケージのインストール
pip install pytest pytest-mock
```

### 全テストの実行

```bash
# プロジェクトルートで実行
python run_tests.py
```

### 個別のテスト実行

```bash
# ユニットテストのみ
pytest tests/inference/utils/test_video_writer.py -v

# 統合テストのみ
pytest tests/inference/test_inference_video_output.py -v

# CLIテストのみ
pytest tests/test_predict_scripts.py -v
```

### クイックチェック（テストなし）

```bash
# ファイルの存在と基本的な構成をチェック
python run_tests.py --quick
```

## テストカバレッジ

### ✅ テスト済み機能

1. **動画書き込み機能**
   - 予測結果の動画への重ね合わせ
   - フレームレートの変換処理
   - カスタムカラー設定
   - フォントサイズ調整

2. **エラーハンドリング**
   - 存在しない動画ファイル
   - 不正な入力データ
   - 書き込み権限エラー

3. **バッチ処理**
   - 複数動画の一括処理
   - CSVファイルからの読み込み

4. **CLIインターフェース**
   - 単一動画の処理
   - バッチ処理
   - ヘルプメッセージ

### ⚠️ モックされている部分

- 実際のモデル読み込み（重いため）
- 実際の推論処理（GPUが必要なため）
- OpenCVの一部機能（CI環境での実行を考慮）

## トラブルシューティング

### ImportError: No module named 'omegaconf'

```bash
pip install omegaconf
```

### ImportError: No module named 'cv2'

```bash
pip install opencv-python
```

### テストが遅い場合

統合テストをスキップ：
```bash
pytest -m "not integration"
```

## 実際の動作確認

テストに加えて、実際の動画で動作確認を行う場合：

```bash
# 単一動画でテスト
python scripts/predict.py input/videos/test.mp4 \
    -o test_predictions.csv \
    --video-output test_labeled.mp4

# 生成された動画を確認
# test_labeled.mp4 に予測結果が重ねられた動画が出力される
```

## 期待される出力

正常に動作した場合、以下のような動画が生成されます：

- 元の動画と同じフレームレート
- 左上に「left: [アクション名]」（赤文字）
- その下に「right: [アクション名]」（緑文字）
- 10Hz（デフォルト）で予測結果が更新

## 今後の改善点

1. パフォーマンステストの追加
2. 異なる動画フォーマットのテスト
3. メモリ使用量のテスト
4. 並列処理のテスト