# KEIBA EDGE NAR版

地方競馬（NAR）に特化したAI競馬予想システム。

JRA版 [keiba-predictor](https://github.com/yssohrrs206038s-oss/keiba-predictor) をベースに、NARデータで学習・予測を行います。

## 概要

- **XGBoost** による3着以内確率予測（NARデータで学習）
- **Claude AI** による自然言語解説生成
- **SHAP値** による予測根拠の可視化
- Discord / X（Twitter）への自動通知
- ダッシュボードでの可視化

## セットアップ

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

## 使い方

```bash
# データ収集（NAR）
python -m keiba_predictor.main scrape --start 2024-01 --end 2026-03 --nar

# クリーニング・特徴量生成
python -m keiba_predictor.main clean
python -m keiba_predictor.main features --league nar

# モデル学習（デフォルトでNAR）
python -m keiba_predictor.main train

# 予測
python -m keiba_predictor.main predict <race_id>
```

## モデルファイルの管理

NARモデル（xgb_model.pkl）は130MB超のためGitHub管理外です。
GitHub Actionsでは Google Drive から自動ダウンロードします。

### セットアップ手順

1. ローカルで学習したモデルを Google Drive にアップロード
2. 共有設定を「リンクを知っている全員」に変更
3. 共有URLからファイルIDを取得:
   `https://drive.google.com/file/d/{FILE_ID}/view` の `{FILE_ID}` 部分
4. GitHub Secrets に登録:
   - `NAR_MODEL_GDRIVE_ID`: Google DriveのファイルID

### 手動ダウンロード

```bash
export NAR_MODEL_GDRIVE_ID="xxxxxxxxxxxxxxxxxxxxx"
python keiba_predictor/scripts/download_model.py
```

## JRA版との違い

| 項目 | JRA版 | NAR版 |
|---|---|---|
| 学習データ | JRAのみ | NARのみ |
| デフォルトleague | `jra` | `nar` |
| 対象レース | 中央競馬重賞 | 地方競馬重賞 |

## ライセンス

本予想はAIによる分析です。馬券購入は自己責任でお願いします。
