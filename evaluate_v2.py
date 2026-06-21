"""
特徴量効果測定 v2
1. 全特徴量AUC
2. オッズ盲目AUC
3. 盲目モデルの◎複勝率 & ◎が1番人気の割合
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("keiba_predictor/data")
from keiba_predictor.features.feature_engineering import FEATURE_COLS

ODDS_COLS = {"odds", "popularity", "prev_odds", "prev2_odds", "prev3_odds"}

df = pd.read_csv(DATA_DIR / "featured_races.csv", parse_dates=["race_date"])
print(f"Loaded: {len(df)} rows, date: {df['race_date'].min().date()} ~ {df['race_date'].max().date()}")

CUTOFF = pd.Timestamp("2024-07-01")
train = df[df["race_date"] < CUTOFF].copy()
test  = df[df["race_date"] >= CUTOFF].copy()
print(f"train: {len(train)}, test: {len(test)}")

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
    # 全テストに確率を書き戻す（mask_teのみ）
    prob_series = pd.Series(np.nan, index=test_df.index)
    prob_series.iloc[np.where(mask_te)[0]] = pred
    imp = pd.Series(model.feature_importances_, index=available).sort_values(ascending=False)
    return auc, prob_series, imp

print("\n=== 1. 全特徴量 AUC ===")
auc_full, _, _ = run_lgb(train, test, FEATURE_COLS, "全特徴量")

print("\n=== 2. オッズ盲目 AUC ===")
blind_cols = [c for c in FEATURE_COLS if c not in ODDS_COLS]
auc_blind, prob_blind, imp_blind = run_lgb(train, test, blind_cols, "オッズ盲目")

# ── 3. 盲目モデルの◎複勝率 & 1番人気割合 ───────────────────────
print("\n=== 3. 盲目モデル ◎複勝率 & ◎が1番人気の割合 ===")
test2 = test.copy()
test2["_prob"] = prob_blind.values

# race_idごとに最高確率の馬を◎とする
test2 = test2[test2["_prob"].notna()].copy()
test2["_rank"] = test2.groupby("race_id")["_prob"].rank(ascending=False, method="first")
honmei = test2[test2["_rank"] == 1].copy()

total_races = honmei["race_id"].nunique()
fukusho_cnt = pd.to_numeric(honmei["top3"], errors="coerce").sum()
fukusho_rate = fukusho_cnt / len(honmei) * 100

# popularityが1(1番人気)の割合
if "popularity" in honmei.columns:
    pop1 = (pd.to_numeric(honmei["popularity"], errors="coerce") == 1).sum()
    pop1_rate = pop1 / len(honmei) * 100
else:
    pop1_rate = float("nan")

print(f"  対象レース数: {total_races}")
print(f"  ◎複勝率: {fukusho_rate:.1f}%  (前回: 34%)")
print(f"  ◎が1番人気の割合: {pop1_rate:.1f}%  (前回: 9.5%)")

# popularity分布
if "popularity" in honmei.columns:
    pop_dist = pd.to_numeric(honmei["popularity"], errors="coerce").value_counts().sort_index().head(10)
    print(f"\n  ◎の人気分布(上位10):")
    for pop, cnt in pop_dist.items():
        print(f"    {int(pop)}番人気: {cnt}頭 ({cnt/len(honmei)*100:.1f}%)")

print("\n=== 盲目モデル 重要度 Top20 ===")
for i, (feat, val) in enumerate(imp_blind.head(20).items(), 1):
    print(f"  {i:2d}. {feat}: {val}")

print("\n完了")
