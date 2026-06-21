"""
ROI感度分析 — 複勝オッズ係数 k=0.30/0.25/0.20
"""
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

DATA_DIR = Path("keiba_predictor/data")
from keiba_predictor.features.feature_engineering import FEATURE_COLS

ODDS_COLS = {"odds", "popularity", "prev_odds", "prev2_odds", "prev3_odds"}

df = pd.read_csv(DATA_DIR / "featured_races.csv", parse_dates=["race_date"], low_memory=False)
CUTOFF = pd.Timestamp("2024-07-01")
train = df[df["race_date"] < CUTOFF].copy()
test  = df[df["race_date"] >= CUTOFF].copy()

blind_cols = [c for c in FEATURE_COLS if c not in ODDS_COLS]
available  = [c for c in blind_cols if c in train.columns]

X_tr = train[available].apply(pd.to_numeric, errors="coerce")
y_tr = pd.to_numeric(train["top3"], errors="coerce")
X_te = test[available].apply(pd.to_numeric, errors="coerce")
y_te = pd.to_numeric(test["top3"], errors="coerce")
mask_tr, mask_te = y_tr.notna(), y_te.notna()

model = lgb.LGBMClassifier(
    objective="binary", metric="auc", verbosity=-1,
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(X_tr[mask_tr], y_tr[mask_tr])
pred_all = model.predict_proba(X_te[mask_te])[:, 1]

test = test[mask_te].copy().reset_index(drop=True)
test["_prob"] = pred_all
test["_rank"] = test.groupby("race_id")["_prob"].rank(ascending=False, method="first")
test["popularity"] = pd.to_numeric(test["popularity"], errors="coerce")
test["odds"]       = pd.to_numeric(test["odds"], errors="coerce")
test["top3"]       = pd.to_numeric(test["top3"], errors="coerce")

head_cnt = test.groupby("race_id")["horse_number"].count().rename("_n_horses")
test = test.join(head_cnt, on="race_id")

honmei = test[test["_rank"] == 1].copy()
honmei["is_pop1"] = (honmei["popularity"] == 1)
diff = honmei[~honmei["is_pop1"]].copy()  # ◎≠1番人気 12,604件

def fuku_odds(row, k):
    if pd.isna(row["odds"]) or pd.isna(row["_n_horses"]):
        return np.nan
    return 1 + (row["odds"] - 1) * k

def roi_stats(sub, k):
    fo = sub.apply(lambda r: fuku_odds(r, k), axis=1)
    hits = sub["top3"] == 1
    n = len(sub)
    ret = (fo[hits] * 100).sum()
    return n, hits.sum(), ret / (n * 100) * 100

def period_label(d):
    if d < pd.Timestamp("2025-01-01"): return "2024後半"
    if d < pd.Timestamp("2025-07-01"): return "2025前半"
    return "2025後半"

KS = [0.30, 0.25, 0.20]

# ── 全体（◎≠1番人気）期間別 ─────────────────────────────────
print("="*70)
print("A. ◎≠1番人気 全12,604レース — 期間別ROI (k別)")
print("="*70)
diff["_period"] = diff["race_date"].map(period_label)
periods = ["2024後半", "2025前半", "2025後半", "全体"]

header = f"  {'期間':<10}" + "".join(f"  k={k:.2f}  " for k in KS) + "  n    複勝率"
print(header)
print("  " + "-"*65)
for period in periods:
    sub = diff if period == "全体" else diff[diff["_period"] == period]
    row = f"  {period:<10}"
    for k in KS:
        n, h, roi = roi_stats(sub, k)
        row += f"  {roi:6.1f}%  "
    fr = sub["top3"].mean() * 100
    row += f"  {len(sub):5d}  {fr:.1f}%"
    print(row)

# ── 単勝3〜10倍帯 ──────────────────────────────────────────
print()
print("="*70)
print("B. ◎≠1番人気 かつ 単勝3〜10倍 — 期間別ROI (k別)")
print("="*70)
band = diff[(diff["odds"] >= 3) & (diff["odds"] < 10)].copy()
band["_period"] = band["race_date"].map(period_label)

print(header)
print("  " + "-"*65)
for period in periods:
    sub = band if period == "全体" else band[band["_period"] == period]
    row = f"  {period:<10}"
    for k in KS:
        n, h, roi = roi_stats(sub, k)
        row += f"  {roi:6.1f}%  "
    fr = sub["top3"].mean() * 100
    row += f"  {len(sub):5d}  {fr:.1f}%"
    print(row)

# ── 月別ROI (k=0.20のみ、◎≠1番人気 全体) ─────────────────
print()
print("="*70)
print("C. 月別ROI — k=0.20 (◎≠1番人気)")
print("="*70)
diff["_ym"] = diff["race_date"].dt.to_period("M")
monthly = []
for ym, sub in diff.groupby("_ym"):
    n, h, roi = roi_stats(sub, 0.20)
    fr = sub["top3"].mean() * 100
    monthly.append({"月": str(ym), "n": n, "複勝率": fr, "ROI(k=0.20)": roi})
mdf = pd.DataFrame(monthly)
print(mdf.to_string(index=False, float_format="%.1f"))

# ── 月別ROI (k=0.20、3〜10倍帯) ────────────────────────────
print()
print("="*70)
print("D. 月別ROI — k=0.20 (3〜10倍帯)")
print("="*70)
band["_ym"] = band["race_date"].dt.to_period("M")
monthly2 = []
for ym, sub in band.groupby("_ym"):
    n, h, roi = roi_stats(sub, 0.20)
    fr = sub["top3"].mean() * 100
    monthly2.append({"月": str(ym), "n": n, "複勝率": fr, "ROI(k=0.20)": roi})
mdf2 = pd.DataFrame(monthly2)
print(mdf2.to_string(index=False, float_format="%.1f"))

print("\n完了")
