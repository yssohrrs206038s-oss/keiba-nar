"""
診断スクリプト: build_features経路 vs live_features経路の精度比較
および train-serve skew の定量化

使い方:
    python keiba_predictor/scripts/diagnose_skew.py

出力:
    1. build_features 経路での 2025-06 sprint 複勝率 (ユーザー数字の再現確認)
    2. live_features 経路シミュレーション (同期間を live と同じ条件に変換)
    3. train-serve skew の feature 別寄与度
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
# ユーティリティ
# ══════════════════════════════════════════════════════════════

def load_model(name: str) -> tuple:
    path = MODEL_DIR / name
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle.get("feature_cols", FEATURE_COLS)

def predict_df(df: pd.DataFrame, model, feature_cols: list) -> pd.Series:
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index)

def metrics(df: pd.DataFrame, prob_col: str, label: str) -> dict:
    """
    レースごとに確率最大馬を◎とし、複勝率を集計。
    1番人気の複勝率と比較。
    """
    races = []
    for race_id, g in df.groupby("race_id"):
        g = g.copy()
        if g[prob_col].isna().all():
            continue
        # ◎ = prob 最大馬
        best_idx = g[prob_col].idxmax()
        hon_top3  = int(g.loc[best_idx, "top3"]) if pd.notna(g.loc[best_idx, "top3"]) else 0
        hon_is_1pop = 1 if pd.to_numeric(g.loc[best_idx, "popularity"], errors="coerce") == 1 else 0

        # 1番人気
        pop1 = g[pd.to_numeric(g["popularity"], errors="coerce") == 1]
        pop1_top3 = int(pop1["top3"].iloc[0]) if (len(pop1) > 0 and pd.notna(pop1["top3"].iloc[0])) else 0

        races.append({
            "hon_top3": hon_top3,
            "hon_is_1pop": hon_is_1pop,
            "pop1_top3": pop1_top3,
        })

    if not races:
        return {}
    rdf = pd.DataFrame(races)
    result = {
        "label": label,
        "n_races": len(rdf),
        "◎複勝率":         f"{rdf['hon_top3'].mean():.1%}",
        "1番人気複勝率":    f"{rdf['pop1_top3'].mean():.1%}",
        "◎が1番人気の割合": f"{rdf['hon_is_1pop'].mean():.1%}",
        "_hon_fukusho":    rdf["hon_top3"].mean(),
        "_pop1_fukusho":   rdf["pop1_top3"].mean(),
        "_hon_is_1pop":    rdf["hon_is_1pop"].mean(),
    }
    return result

def print_result(r: dict) -> None:
    print(f"\n{'─'*50}")
    print(f"  {r['label']}  (n={r['n_races']}レース)")
    print(f"{'─'*50}")
    print(f"  ◎複勝率         : {r['◎複勝率']}")
    print(f"  1番人気複勝率   : {r['1番人気複勝率']}")
    print(f"  ◎が1番人気の割合: {r['◎が1番人気の割合']}")

# ══════════════════════════════════════════════════════════════
# 1. build_features 経路 (ユーザーの計測と同条件)
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("  KEIBA NAR 診断: build_features vs live 経路比較")
print("=" * 60)

print("\n[1/4] cleaned_races.csv.gz を読み込み + build_features() 実行中...")
gz = DATA_DIR / "cleaned_races.csv.gz"
raw = pd.read_csv(gz, encoding="utf-8-sig", parse_dates=["race_date"])
print(f"      全データ: {len(raw):,} 行 / {raw['race_date'].min().date()} ~ {raw['race_date'].max().date()}")

featured = build_features(raw)

# 2025-06 sprint フィルタ
target = featured[
    (featured["race_date"] >= "2025-06-01") &
    (featured["race_date"] <  "2025-07-01") &
    (pd.to_numeric(featured["distance"], errors="coerce") < 1400)
].copy()

print(f"      2025-06 sprint データ: {len(target):,} 行 / {target['race_id'].nunique()} レース")
if len(target) == 0:
    print("      ⚠️  対象データなし → 終了")
    sys.exit(1)

# xgb_model_sprint.pkl で予測
print("\n[2/4] xgb_model_sprint.pkl で予測 (build_features 経路)...")
model_sprint, feat_cols_sprint = load_model("xgb_model_sprint.pkl")
target["prob_build"] = predict_df(target, model_sprint, feat_cols_sprint)

r_build = metrics(target, "prob_build", "build_features 経路 (ユーザー計測と同条件)")
print_result(r_build)

# ══════════════════════════════════════════════════════════════
# 2. live_features 経路シミュレーション
#    build_features 済みデータを live と同じ条件に変換する
# ══════════════════════════════════════════════════════════════

print("\n[3/4] live_features 経路シミュレーション...")
print("      (同じ特徴量データを live と同じ NaN パターンに変換して再予測)")

sim = target.copy()

# ── live では常に NaN になる特徴量 ──────────────────────────
# last_3f : 当日レース結果なので live では未知
sim["last_3f"] = np.nan
# weather_enc : live の build_live_features では np.nan で固定
sim["weather_enc"] = np.nan

# ── キー名バグ: jockey_district_fukusho_rate が live では median 補完される ─
# build_live_features の row dict (live_features.py:511) には
#   "jockey_course_fukusho_rate": jockey_course_rate
# と書かれており、FEATURE_COLS の "jockey_district_fukusho_rate" と名前が異なる。
# そのため live では median 補完される (全馬同じ値になる)。
median_jd = pd.to_numeric(featured["jockey_district_fukusho_rate"], errors="coerce").median()
sim["jockey_district_fukusho_rate"] = median_jd  # 全馬一律 → 識別力ゼロ

# NaN → median 補完 (live の build_live_features と同じ処理)
# (same_day_rank, prob_vs_avg は build_features でも NaN なので変化なし)
for col in FEATURE_COLS:
    if col not in sim.columns:
        continue
    median_val = pd.to_numeric(featured[col], errors="coerce").median() if col in featured.columns else np.nan
    null_mask = pd.to_numeric(sim[col], errors="coerce").isna()
    sim.loc[null_mask, col] = median_val

sim["prob_live"] = predict_df(sim, model_sprint, feat_cols_sprint)

r_live = metrics(sim, "prob_live", "live_features 経路シミュレーション")
print_result(r_live)

# ══════════════════════════════════════════════════════════════
# 3. 差分サマリ
# ══════════════════════════════════════════════════════════════

print("\n[4/4] train-serve skew 定量化")
print("      各特徴量を1つずつ live 化して◎複勝率の変化を測定...")

SKEW_FEATURES = {
    "last_3f":                    ("当日上がり3F", "常に NaN (live では結果なし)"),
    "weather_enc":                ("天候エンコード", "常に NaN (live ではスクレイピング非対応)"),
    "jockey_district_fukusho_rate":("騎手×地区複勝率", "キー名バグ → median補完 (全馬同値で識別力0)"),
    "same_day_rank":              ("同日内AI確率順位", "訓練でも NaN (学習/推論で一貫してNaN)"),
    "prob_vs_avg":                ("AI確率-平均差", "訓練でも NaN (学習/推論で一貫してNaN)"),
}

base_fukusho = r_build["_hon_fukusho"]
print(f"\n  ベースライン (build 経路): ◎複勝率 = {base_fukusho:.1%}")
print(f"  {'特徴量':<35} {'live化後の複勝率':>12} {'差分':>8}  説明")
print(f"  {'─'*35} {'─'*12} {'─'*8}  {'─'*40}")

for feat, (ja_name, note) in SKEW_FEATURES.items():
    if feat not in target.columns:
        print(f"  {ja_name:<35} {'N/A':>12} {'N/A':>8}  {note}")
        continue
    tmp = target.copy()
    if feat == "jockey_district_fukusho_rate":
        tmp[feat] = median_jd
    elif feat in ("same_day_rank", "prob_vs_avg"):
        pass  # すでに NaN → 変化なし
    else:
        tmp[feat] = np.nan
        # median 補完
        mv = pd.to_numeric(featured[feat], errors="coerce").median()
        tmp[feat] = mv

    tmp["prob_tmp"] = predict_df(tmp, model_sprint, feat_cols_sprint)
    r_tmp = metrics(tmp, "prob_tmp", "")
    delta = r_tmp["_hon_fukusho"] - base_fukusho
    sign = "+" if delta >= 0 else ""
    print(f"  {ja_name:<35} {r_tmp['_hon_fukusho']:>12.1%} {sign}{delta:>7.1%}  {note}")

# ══════════════════════════════════════════════════════════════
# 4. 結論
# ══════════════════════════════════════════════════════════════

print(f"""
{'='*60}
  診断サマリ
{'='*60}

  build_features 経路  ◎複勝率: {r_build['_hon_fukusho']:.1%}
  live_features  経路  ◎複勝率: {r_live['_hon_fukusho']:.1%}
  差分 (live - build)        : {r_live['_hon_fukusho'] - r_build['_hon_fukusho']:+.1%}

  1番人気複勝率 (参考)       : {r_build['_pop1_fukusho']:.1%}
  ◎が1番人気の割合 (build)   : {r_build['_hon_is_1pop']:.1%}
  ◎が1番人気の割合 (live)    : {r_live['_hon_is_1pop']:.1%}

  【elo_minus_field_avg / is_jockey_change について】
  これらの特徴量は現在の FEATURE_COLS に存在しません。
  コードベース全体を検索しても定義・参照箇所がありません。

  【実際の train-serve skew 源泉 (上記テーブル参照)】
  1. jockey_district_fukusho_rate: live_features.py:511 でキー名が
     "jockey_course_fukusho_rate" と書かれており FEATURE_COLS の名前と不一致。
     → live では全馬が同じ median 値 ({median_jd:.3f}) になり識別力ゼロ。
  2. last_3f: 当日の上がり3Fは live では NaN → median補完。
     訓練データには実績値があるため leakage かつ skew の源泉。
  3. weather_enc: live では NaN → median補完。
  4. same_day_rank / prob_vs_avg: 訓練でも NaN のため skew なし。

{'='*60}
""")
