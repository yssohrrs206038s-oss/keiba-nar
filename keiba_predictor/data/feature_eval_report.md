# 特徴量追加効果測定レポート

生成日: 2026-06-13 00:14
訓練期間: ～ 2024-07-01 / 検証期間: 2024-07-01 ～ 2026-01-01
訓練行数: 528,330 / 検証行数: 237,966

## AUC 比較

| モデル | 特徴量数 | Val AUC |
|--------|----------|---------|
| 旧     | 47 | 0.8216 |
| 新     | 56 | 0.8232 |
| 差     |          | +0.0016 |

## ブラインドモデル ROI 比較（単勝1点買い、Val期間）

| モデル | ROI  |
|--------|------|
| 旧     | 0.741 |
| 新     | 0.774 |
| 差     | +0.033 |

## XGBoost 特徴量重要度 Top 20（新モデル）

| Rank | Feature | Gain |
|------|---------|------|
| 1 | last_3f | 786.1 |
| 2 | odds | 724.6 |
| 3 | popularity | 232.3 |
| 4 | elo_minus_field_avg ★ | 18.0 |
| 5 | is_continuous | 16.2 |
| 6 | is_fresh | 15.2 |
| 7 | sire_win_rate | 15.0 |
| 8 | days_since_last_race | 12.7 |
| 9 | bms_win_rate | 12.0 |
| 10 | horse_weight | 11.9 |
| 11 | weeks_since_last_race | 11.8 |
| 12 | prev_finish_pos | 11.5 |
| 13 | horse_weight_diff | 10.6 |
| 14 | prev2_odds | 10.3 |
| 15 | sire_dist_win_rate | 9.9 |
| 16 | prev2_finish_pos | 9.9 |
| 17 | horse_number | 9.7 |
| 18 | prev_odds | 9.7 |
| 19 | frame_number | 9.6 |
| 20 | prev2_last_3f | 9.5 |

（★ = 今回追加した新特徴量）

## 新特徴量のみの重要度

| Feature | Gain |
|---------|------|
| elo_minus_field_avg | 18.0 |
