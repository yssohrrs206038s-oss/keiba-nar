"""
モデルの付加価値定量化スクリプト

「◎が1番人気以外のレース」に絞り、
モデルが市場の見落としを発見できているかを検証する。

対象: 2025年全期間・全距離帯
経路: live_features シミュレーション
"""

import sys
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from keiba_predictor.features.feature_engineering import FEATURE_COLS, build_features

DATA_DIR  = ROOT / "keiba_predictor" / "data"
MODEL_DIR = ROOT / "keiba_predictor" / "model"

# ══════════════════════════════════════════════════════════════
# モデル読み込み・予測
# ══════════════════════════════════════════════════════════════

def load_model(name: str):
    path = MODEL_DIR / name
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle.get("feature_cols", FEATURE_COLS)

# 距離帯別モデルをロード
BAND_MODELS = {}
for band in ("sprint", "mile", "middle", "long"):
    m, fc = load_model(f"xgb_model_{band}.pkl")
    if m is not None:
        BAND_MODELS[band] = (m, fc)

def classify_band(dist):
    d = float(dist) if pd.notna(dist) else 0
    if d < 1400:   return "sprint"
    if d <= 1800:  return "mile"
    if d <= 2200:  return "middle"
    return "long"

def predict_by_band(df: pd.DataFrame) -> pd.Series:
    """距離帯ごとに適切なモデルで予測し、全行分の prob を返す。"""
    probs = pd.Series(np.nan, index=df.index)
    df = df.copy()
    df["_band"] = pd.to_numeric(df["distance"], errors="coerce").apply(classify_band)
    for band, (model, feat_cols) in BAND_MODELS.items():
        mask = df["_band"] == band
        if not mask.any():
            continue
        sub = df.loc[mask]
        cols = [c for c in feat_cols if c in sub.columns]
        X = sub[cols].apply(pd.to_numeric, errors="coerce")
        probs.loc[mask] = model.predict_proba(X)[:, 1]
    return probs

# ══════════════════════════════════════════════════════════════
# live シミュレーション変換
# ══════════════════════════════════════════════════════════════

def apply_live_sim(df: pd.DataFrame, full_featured: pd.DataFrame) -> pd.DataFrame:
    """build_features 済みデータを live 条件に変換する。"""
    sim = df.copy()
    # live では常に NaN
    sim["last_3f"]     = np.nan
    sim["weather_enc"] = np.nan
    # キー名バグ: jockey_district_fukusho_rate → median 補完
    median_jd = pd.to_numeric(full_featured["jockey_district_fukusho_rate"], errors="coerce").median()
    sim["jockey_district_fukusho_rate"] = median_jd
    # 残りの NaN → median 補完
    for col in FEATURE_COLS:
        if col not in sim.columns:
            continue
        mv = pd.to_numeric(full_featured[col], errors="coerce").median() if col in full_featured.columns else np.nan
        null_mask = pd.to_numeric(sim[col], errors="coerce").isna()
        sim.loc[null_mask, col] = mv
    return sim

# ══════════════════════════════════════════════════════════════
# メイン
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("  モデル付加価値診断: ◎≠1番人気レースの精度検証")
print("=" * 60)

print("\n[1/4] データ読み込み + build_features() ...")
gz = DATA_DIR / "cleaned_races.csv.gz"
raw = pd.read_csv(gz, encoding="utf-8-sig", parse_dates=["race_date"])
print(f"      全データ: {len(raw):,} 行 ({raw['race_date'].min().date()} ~ {raw['race_date'].max().date()})")

featured_all = build_features(raw)

# 2025年全期間フィルタ
target = featured_all[
    (featured_all["race_date"] >= "2025-01-01") &
    (featured_all["race_date"] <  "2026-01-01")
].copy()
print(f"      2025年データ: {len(target):,} 行 / {target['race_id'].nunique():,} レース")

# live シミュレーション適用
print("\n[2/4] live_features シミュレーション適用 ...")
sim = apply_live_sim(target, featured_all)

# 予測
print("[3/4] 距離帯別モデルで予測 ...")
sim["prob"] = predict_by_band(sim)

# ── レース単位で集計 ─────────────────────────────────────────
print("[4/4] 集計 ...")

records = []
for race_id, g in sim.groupby("race_id"):
    if g["prob"].isna().all():
        continue
    pop_num = pd.to_numeric(g["popularity"], errors="coerce")
    top3_num = pd.to_numeric(g["top3"], errors="coerce")
    odds_num  = pd.to_numeric(g["odds"], errors="coerce")

    # ◎
    best_idx = g["prob"].idxmax()
    hon_top3    = int(top3_num.loc[best_idx]) if pd.notna(top3_num.loc[best_idx]) else None
    hon_pop     = pop_num.loc[best_idx]
    hon_odds    = odds_num.loc[best_idx]
    hon_is_1pop = (hon_pop == 1)

    # 1番人気
    pop1 = g[pop_num == 1]
    pop1_top3 = int(top3_num.loc[pop1.index[0]]) if (len(pop1) > 0 and pd.notna(top3_num.loc[pop1.index[0]])) else None

    records.append({
        "race_id":      race_id,
        "hon_top3":     hon_top3,
        "hon_pop":      hon_pop,
        "hon_odds":     hon_odds,
        "hon_is_1pop":  hon_is_1pop,
        "pop1_top3":    pop1_top3,
    })

rdf = pd.DataFrame(records)
rdf = rdf.dropna(subset=["hon_top3", "pop1_top3"])

n_total = len(rdf)
n_1pop  = rdf["hon_is_1pop"].sum()
n_not1  = n_total - n_1pop

print(f"\n      全レース: {n_total:,} / うち◎=1番人気: {n_1pop:,} ({n_1pop/n_total:.1%}) / ◎≠1番人気: {n_not1:,} ({n_not1/n_total:.1%})")

# ══════════════════════════════════════════════════════════════
# 1. 全体サマリ
# ══════════════════════════════════════════════════════════════

all_hon_f   = rdf["hon_top3"].mean()
all_pop1_f  = rdf["pop1_top3"].mean()

sub = rdf[~rdf["hon_is_1pop"]]
sub_hon_f  = sub["hon_top3"].mean()
sub_pop1_f = sub["pop1_top3"].mean()

print(f"""
{'='*60}
  集計結果
{'='*60}

  【全レース ({n_total:,}レース)】
    ◎複勝率        : {all_hon_f:.1%}
    1番人気複勝率   : {all_pop1_f:.1%}

  【◎ = 1番人気 ({n_1pop:,}レース, {n_1pop/n_total:.1%})】
    ◎複勝率        : {rdf[rdf['hon_is_1pop']]['hon_top3'].mean():.1%}
    1番人気複勝率   : {rdf[rdf['hon_is_1pop']]['pop1_top3'].mean():.1%}

  【◎ ≠ 1番人気 ({n_not1:,}レース, {n_not1/n_total:.1%})】  ← 付加価値の本丸
    ◎複勝率        : {sub_hon_f:.1%}  ← これが1番人気複勝率を上回れば価値あり
    1番人気複勝率   : {sub_pop1_f:.1%}  ← 同レース内の1番人気成績
    差分 (◎ - 1人気): {sub_hon_f - sub_pop1_f:+.1%}
""")

# ══════════════════════════════════════════════════════════════
# 2. オッズ帯別 (◎ ≠ 1番人気のみ)
# ══════════════════════════════════════════════════════════════

def odds_band(o):
    if pd.isna(o):    return "不明"
    if o < 3:         return "~3倍"   # 2番人気付近
    if o < 5:         return "3~5倍"
    if o < 10:        return "5~10倍"
    if o < 20:        return "10~20倍"
    return "20倍~"

sub = sub.copy()
sub["odds_band"] = sub["hon_odds"].apply(odds_band)

BAND_ORDER = ["~3倍", "3~5倍", "5~10倍", "10~20倍", "20倍~", "不明"]
print("  【オッズ帯別 (◎ ≠ 1番人気のみ)】")
print(f"  {'オッズ帯':<12} {'n':>6} {'◎複勝率':>10} {'1人気複勝率':>12} {'差分':>8}")
print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*12} {'─'*8}")

for band in BAND_ORDER:
    b = sub[sub["odds_band"] == band]
    if len(b) == 0:
        continue
    h = b["hon_top3"].mean()
    p = b["pop1_top3"].mean()
    marker = " ✓" if h > p else ""
    print(f"  {band:<12} {len(b):>6,} {h:>10.1%} {p:>12.1%} {h-p:>+8.1%}{marker}")

# ══════════════════════════════════════════════════════════════
# 3. 距離帯別 (◎ ≠ 1番人気のみ)
# ══════════════════════════════════════════════════════════════

sub2 = sim[~sim["race_id"].isin(rdf[rdf["hon_is_1pop"]]["race_id"])].copy()

print(f"\n  【距離帯別 (◎ ≠ 1番人気のみ)】")
print(f"  {'距離帯':<12} {'対象レース':>10} {'◎複勝率':>10} {'1人気複勝率':>12} {'差分':>8}")
print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*8}")

sub_races = sub[["race_id","hon_top3","pop1_top3","hon_is_1pop"]].copy()

dist_df = (
    sim[["race_id","distance"]]
    .drop_duplicates("race_id")
    .assign(band=lambda d: pd.to_numeric(d["distance"], errors="coerce").apply(classify_band))
)
sub_with_band = sub_races.merge(dist_df, on="race_id", how="left")

for band in ("sprint","mile","middle","long"):
    b = sub_with_band[sub_with_band["band"] == band]
    if len(b) == 0:
        continue
    h = b["hon_top3"].mean()
    p = b["pop1_top3"].mean()
    marker = " ✓" if h > p else ""
    dist_label = {"sprint":"短距離<1400","mile":"マイル1400-1800",
                  "middle":"中距離1800-2200","long":"長距離>2200"}[band]
    print(f"  {dist_label:<18} {len(b):>6,} {h:>10.1%} {p:>12.1%} {h-p:>+8.1%}{marker}")

print(f"\n{'='*60}")
print("  ✓ = ◎複勝率 > 同レース1番人気複勝率 (市場見落とし発見)")
print(f"{'='*60}\n")
