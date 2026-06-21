"""
新特徴量の効果測定スクリプト
- NaN率確認
- 時系列分割でAUC比較(旧vs新)
- オッズ盲目AUC比較
- 重要度Top20
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("keiba_predictor/data")

from keiba_predictor.features.feature_engineering import FEATURE_COLS

ODDS_COLS = {"odds", "popularity", "prev_odds", "prev2_odds", "prev3_odds"}

OLD_FEATURE_COLS = [
    "distance", "course_type_enc", "track_condition_enc", "weather_enc",
    "frame_number", "horse_number", "weight_carried", "odds", "popularity",
    "sex_enc", "age", "horse_weight", "horse_weight_diff",
    "avg_time_3", "avg_time_5", "avg_time_3_any", "avg_time_5_any",
    "jockey_fukusho_rate", "trainer_fukusho_rate", "dist_diff_prev",
    "days_since_last_race", "prev_finish_pos", "prev_odds",
    "prev2_finish_pos", "prev2_odds", "prev3_finish_pos",
    "prev2_last_3f", "prev3_last_3f", "finish_pos_trend",
    "horse_course_fukusho_rate", "horse_dist_fukusho_rate",
    "jockey_horse_fukusho_rate", "horse_track_fukusho_rate",
    "jockey_district_fukusho_rate", "jockey_dist_fukusho_rate",
    "weeks_since_last_race", "is_fresh", "is_continuous",
    "jockey_trainer_fukusho_rate", "weight_carried_diff", "is_weight_increase",
    "same_day_rank", "prob_vs_avg",
    "sire_win_rate", "bms_win_rate", "sire_course_win_rate",
    "sire_dist_win_rate", "bms_course_win_rate",
    "race_class_level", "prev_class_level", "class_diff", "is_class_up", "is_class_down",
    "horse_elo", "elo_minus_field_avg", "prev_race_opp_elo",
    "is_jockey_change",
]

df = pd.read_csv(DATA_DIR / "featured_races.csv", parse_dates=["race_date"])
print(f"Loaded: {len(df)} rows, {len(df.columns)} cols")
print(f"Date range: {df['race_date'].min()} ~ {df['race_date'].max()}")

# ── 新特徴量列の確認 ────────────────────────────────────────────
NEW_COLS = [
    "prev_last_3f", "prev_margin",
    "avg_finish_3", "avg_popularity_3", "avg_last3f_3", "avg_margin_3",
    "avg_finish_5", "avg_last3f_5",
    "same_distance_rate", "same_venue_rate", "going_good_rate", "going_bad_rate",
    "jockey_win_rate", "trainer_win_rate", "jockey_venue_rate",
]
if "prev_passing" in df.columns:
    NEW_COLS.append("prev_passing")

print("\n=== 新特徴量 NaN率 ===")
for col in NEW_COLS:
    if col in df.columns:
        nan_pct = df[col].isna().mean() * 100
        flag = " *** 50%超 ***" if nan_pct >= 50 else ""
        print(f"  {col}: {nan_pct:.1f}%{flag}")
    else:
        print(f"  {col}: 列なし")

# ── 時系列分割 ────────────────────────────────────────────────
CUTOFF = pd.Timestamp("2024-07-01")
train = df[df["race_date"] < CUTOFF].copy()
test  = df[df["race_date"] >= CUTOFF].copy()
print(f"\n学習: {len(train)} / テスト: {len(test)}")

def run_lgb(train_df, test_df, feat_cols, label=""):
    available = [c for c in feat_cols if c in train_df.columns]
    X_tr = train_df[available].apply(pd.to_numeric, errors="coerce")
    y_tr = pd.to_numeric(train_df["top3"], errors="coerce")
    X_te = test_df[available].apply(pd.to_numeric, errors="coerce")
    y_te = pd.to_numeric(test_df["top3"], errors="coerce")

    mask_tr = y_tr.notna()
    mask_te = y_te.notna()

    params = dict(
        objective="binary", metric="auc", verbosity=-1,
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr[mask_tr], y_tr[mask_tr])
    pred = model.predict_proba(X_te[mask_te])[:, 1]
    auc = roc_auc_score(y_te[mask_te], pred)
    print(f"  [{label}] AUC={auc:.4f}  (feat={len(available)})")
    return model, auc, available

print("\n=== AUC比較 ===")
_, auc_old, _ = run_lgb(train, test, OLD_FEATURE_COLS, "旧特徴量")
model_new, auc_new, new_available = run_lgb(train, test, FEATURE_COLS, "新特徴量")

# オッズ盲目
old_blind = [c for c in OLD_FEATURE_COLS if c not in ODDS_COLS]
new_blind = [c for c in FEATURE_COLS if c not in ODDS_COLS]
print("\n=== オッズ盲目 AUC ===")
run_lgb(train, test, old_blind, "旧・盲目")
run_lgb(train, test, new_blind, "新・盲目")

# ── 重要度Top20 ──────────────────────────────────────────────
print("\n=== 新特徴量 重要度 Top20 ===")
imp = pd.Series(model_new.feature_importances_, index=new_available).sort_values(ascending=False)
for i, (feat, val) in enumerate(imp.head(20).items(), 1):
    print(f"  {i:2d}. {feat}: {val}")

print("\n完了")
