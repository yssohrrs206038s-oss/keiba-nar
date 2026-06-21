"""
盲目モデル付加価値診断
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("keiba_predictor/data")
from keiba_predictor.features.feature_engineering import FEATURE_COLS

ODDS_COLS = {"odds", "popularity", "prev_odds", "prev2_odds", "prev3_odds"}

df = pd.read_csv(DATA_DIR / "featured_races.csv", parse_dates=["race_date"],
                 low_memory=False)

CUTOFF = pd.Timestamp("2024-07-01")
train = df[df["race_date"] < CUTOFF].copy()
test  = df[df["race_date"] >= CUTOFF].copy()
print(f"train: {len(train)}, test: {len(test)}")

# ── 盲目モデル学習 ──────────────────────────────────────────────
blind_cols = [c for c in FEATURE_COLS if c not in ODDS_COLS]
available  = [c for c in blind_cols if c in train.columns]

X_tr = train[available].apply(pd.to_numeric, errors="coerce")
y_tr = pd.to_numeric(train["top3"], errors="coerce")
X_te = test[available].apply(pd.to_numeric, errors="coerce")
y_te = pd.to_numeric(test["top3"], errors="coerce")
mask_tr = y_tr.notna()
mask_te = y_te.notna()

model = lgb.LGBMClassifier(
    objective="binary", metric="auc", verbosity=-1,
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X_tr[mask_tr], y_tr[mask_tr])
pred_all = model.predict_proba(X_te[mask_te])[:, 1]
auc = roc_auc_score(y_te[mask_te], pred_all)
print(f"盲目AUC: {auc:.4f}")

# テストに確率を書き戻す
test = test[mask_te].copy().reset_index(drop=True)
test["_prob"] = pred_all

# ── ◎選定 (race_id内最高確率) ────────────────────────────────
test["_rank"] = test.groupby("race_id")["_prob"].rank(ascending=False, method="first")
honmei = test[test["_rank"] == 1].copy()
honmei["top3"]       = pd.to_numeric(honmei["top3"], errors="coerce")
honmei["popularity"] = pd.to_numeric(honmei["popularity"], errors="coerce")
honmei["odds"]       = pd.to_numeric(honmei["odds"], errors="coerce")

# 1番人気馬をレースごとに取得
pop1 = test[pd.to_numeric(test["popularity"], errors="coerce") == 1].copy()
pop1["top3"] = pd.to_numeric(pop1["top3"], errors="coerce")
pop1_map = pop1.set_index("race_id")[["top3", "odds"]].rename(
    columns={"top3": "pop1_top3", "odds": "pop1_odds"})

honmei = honmei.join(pop1_map, on="race_id")
honmei["is_pop1"] = (honmei["popularity"] == 1)

# レース内頭数（複勝オッズ計算に必要）
head_cnt = test.groupby("race_id")["horse_number"].count().rename("_n_horses")
honmei = honmei.join(head_cnt, on="race_id")

def fukusho_odds_approx(row):
    """複勝オッズ近似: 1 + (odds-1)*0.30 (9頭以上) / *0.45 (8頭以下)"""
    if pd.isna(row["odds"]) or pd.isna(row["_n_horses"]):
        return np.nan
    rate = 0.30 if row["_n_horses"] >= 9 else 0.45
    return 1 + (row["odds"] - 1) * rate

honmei["_fuku_odds"] = honmei.apply(fukusho_odds_approx, axis=1)

print("\n" + "="*60)
print("1. 全体◎複勝率")
print("="*60)
total = len(honmei)
hits  = honmei["top3"].sum()
print(f"  {hits:.0f} / {total} = {hits/total*100:.1f}%")

print("\n" + "="*60)
print("2. ◎=1番人気 vs ◎≠1番人気")
print("="*60)
for flag, label in [(True, "◎=1番人気"), (False, "◎≠1番人気")]:
    sub = honmei[honmei["is_pop1"] == flag]
    r   = sub["top3"].mean() * 100
    print(f"  {label}: {r:.1f}%  (n={len(sub)})")

print("\n" + "="*60)
print("3. ◎≠1番人気レースの ◎複勝率 vs 1番人気複勝率")
print("="*60)
diff = honmei[~honmei["is_pop1"]].copy()
c_rate = diff["top3"].mean() * 100
p_rate = diff["pop1_top3"].mean() * 100
print(f"  ◎複勝率:    {c_rate:.1f}%  (n={len(diff)})")
print(f"  1番人気複勝率: {p_rate:.1f}%")
print(f"  差: {c_rate - p_rate:+.1f}pt")

print("\n" + "="*60)
print("4. オッズ帯別分析（◎≠1番人気）")
print("="*60)
bins   = [0, 3, 5, 10, 20, 999]
labels = ["~3倍", "3-5倍", "5-10倍", "10-20倍", "20倍~"]
diff["_odds_band"] = pd.cut(diff["odds"], bins=bins, labels=labels, right=True)

print(f"  {'帯':<10} {'n':>5}  {'◎複勝率':>8}  {'1番人気複勝率':>12}  {'差':>6}")
print("  " + "-"*55)
band_rows = []
for band in labels:
    sub = diff[diff["_odds_band"] == band]
    if len(sub) == 0:
        continue
    c = sub["top3"].mean() * 100
    p = sub["pop1_top3"].mean() * 100
    gap = c - p
    flag = " ◀ 優位" if gap > 0 else ""
    print(f"  {band:<10} {len(sub):>5}  {c:>7.1f}%  {p:>11.1f}%  {gap:>+6.1f}pt{flag}")
    band_rows.append({"band": band, "n": len(sub), "c_rate": c, "p_rate": p, "gap": gap, "sub": sub})

print("\n" + "="*60)
print("5. 優位帯でのROI（◎≠1番人気 かつ ◎>1番人気複勝率）")
print("="*60)
winning_bands = [r for r in band_rows if r["gap"] > 0]
if not winning_bands:
    print("  ◎が1番人気を上回る帯なし")
else:
    roi_sub = pd.concat([r["sub"] for r in winning_bands])
    n_races  = len(roi_sub)
    total_cost = n_races * 100
    hit_sub  = roi_sub[roi_sub["top3"] == 1]
    total_return = (hit_sub["_fuku_odds"] * 100).sum()
    roi = total_return / total_cost * 100
    print(f"  対象帯: {[r['band'] for r in winning_bands]}")
    print(f"  購入レース数: {n_races}")
    print(f"  的中数: {len(hit_sub)}  ({len(hit_sub)/n_races*100:.1f}%)")
    print(f"  総投資: {total_cost:,.0f}円")
    print(f"  総回収(近似): {total_return:,.0f}円")
    print(f"  ROI: {roi:.1f}%")

print("\n" + "="*60)
print("6. 年別ROI安定性（優位帯 ◎≠1番人気）")
print("="*60)
def period_label(d):
    if d < pd.Timestamp("2025-01-01"): return "2024後半"
    if d < pd.Timestamp("2025-07-01"): return "2025前半"
    return "2025後半"

diff2 = diff.copy()
if winning_bands:
    winning_band_names = [r["band"] for r in winning_bands]
    roi_sub2 = diff2[diff2["_odds_band"].isin(winning_band_names)].copy()
else:
    roi_sub2 = diff2.copy()

roi_sub2["_period"] = roi_sub2["race_date"].map(period_label)
print(f"  {'期間':<12} {'n':>5}  {'複勝率':>7}  {'ROI':>8}")
print("  " + "-"*40)
for period in ["2024後半", "2025前半", "2025後半"]:
    s = roi_sub2[roi_sub2["_period"] == period]
    if len(s) == 0:
        continue
    hit = s[s["top3"] == 1]
    ret = (hit["_fuku_odds"] * 100).sum()
    r   = ret / (len(s) * 100) * 100
    fr  = s["top3"].mean() * 100
    print(f"  {period:<12} {len(s):>5}  {fr:>6.1f}%  {r:>7.1f}%")

print("\n完了")
