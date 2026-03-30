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

## JRA版との違い

| 項目 | JRA版 | NAR版 |
|---|---|---|
| 学習データ | JRAのみ | NARのみ |
| デフォルトleague | `jra` | `nar` |
| 対象レース | 中央競馬重賞 | 地方競馬重賞 |

## ライセンス

本予想はAIによる分析です。馬券購入は自己責任でお願いします。
