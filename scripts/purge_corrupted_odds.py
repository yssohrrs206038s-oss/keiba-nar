"""
cleaned_races.csv.gz の汚染 odds/popularity をレース単位でNaNに置換する。

汚染条件（2026-04-01以降のレースのみ対象）:
  - レース内 Σ(1/odds) < 1.0（オーバーラウンドが1未満 = オッズ列に別データが混入）
  - または popularity が全行 NaN

行は残す。着順・馬名・タイム等の列は有効なため削除しない。

使い方:
  python scripts/purge_corrupted_odds.py            # dry-run（変更なし）
  python scripts/purge_corrupted_odds.py --apply    # 実際に上書き保存
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR     = REPO_ROOT / "keiba_predictor" / "data"
GZ_PATH      = DATA_DIR / "cleaned_races.csv.gz"
CSV_PATH     = DATA_DIR / "cleaned_races.csv"

# 汚染チェック開始日（この日付以降のレースのみ検査）
CHECK_FROM   = pd.Timestamp("2026-04-01")
# オーバーラウンド閾値: 1.0 未満を異常とみなす
OVERROUND_MIN = 1.0


def _overround(odds_series: pd.Series) -> float:
    """レース内の Σ(1/odds) を返す。有効なオッズがない場合は 0.0。"""
    num = pd.to_numeric(odds_series, errors="coerce").dropna()
    if len(num) == 0:
        return 0.0
    return float((1.0 / num.clip(lower=0.01)).sum())


def detect_corrupted_races(df: pd.DataFrame) -> pd.DataFrame:
    """
    汚染レース（race_id単位）の情報を返す DataFrame。

    カラム: race_id, race_date, n_rows, overround, pop_all_nan, reason
    """
    target = df[df["race_date"] >= CHECK_FROM].copy()
    if target.empty:
        logger.info(f"{CHECK_FROM.date()} 以降のデータがありません。")
        return pd.DataFrame()

    records = []
    for race_id, grp in target.groupby("race_id", sort=False):
        or_val    = _overround(grp["odds"])
        pop_nan   = pd.to_numeric(grp.get("popularity", pd.Series()), errors="coerce").isna().all()
        bad_or    = or_val < OVERROUND_MIN
        reasons   = []
        if bad_or:
            reasons.append(f"overround={or_val:.3f}<{OVERROUND_MIN}")
        if pop_nan:
            reasons.append("popularity全NaN")
        if reasons:
            records.append({
                "race_id":    race_id,
                "race_date":  grp["race_date"].iloc[0],
                "n_rows":     len(grp),
                "overround":  round(or_val, 4),
                "pop_all_nan": pop_nan,
                "reason":     " / ".join(reasons),
            })

    return pd.DataFrame(records)


def purge(apply: bool) -> None:
    path = GZ_PATH if GZ_PATH.exists() else CSV_PATH
    if not path.exists():
        logger.error(f"データファイルが見つかりません: {GZ_PATH} / {CSV_PATH}")
        sys.exit(1)

    logger.info(f"読み込み: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["race_date"], low_memory=False)
    logger.info(f"  総行数: {len(df):,}, 期間: {df['race_date'].min()} 〜 {df['race_date'].max()}")

    corrupted = detect_corrupted_races(df)

    if corrupted.empty:
        logger.info(f"{CHECK_FROM.date()} 以降に汚染レースは見つかりませんでした。")
        return

    n_races = len(corrupted)
    n_rows  = int(corrupted["n_rows"].sum())
    logger.info(f"\n汚染レース数: {n_races:,} レース / {n_rows:,} 行")
    logger.info(f"  うちオーバーラウンド異常: {corrupted['overround'].lt(OVERROUND_MIN).sum():,} レース")
    logger.info(f"  うちpopularity全NaN:     {corrupted['pop_all_nan'].sum():,} レース")

    # 上位20件を表示
    logger.info("\n汚染レース一覧（先頭20件）:")
    for _, row in corrupted.head(20).iterrows():
        logger.info(
            f"  {str(row['race_date'])[:10]}  {row['race_id']}  "
            f"rows={row['n_rows']}  {row['reason']}"
        )

    if not apply:
        logger.info(
            "\ndry-run モードのため変更しません。"
            "--apply を指定すると odds/popularity を NaN に置換して保存します。"
        )
        return

    # 汚染 race_id セット
    bad_ids = set(corrupted["race_id"].astype(str))

    before_odds_valid = pd.to_numeric(df["odds"], errors="coerce").notna().sum()
    before_pop_valid  = pd.to_numeric(df.get("popularity", pd.Series()), errors="coerce").notna().sum()

    mask = df["race_id"].astype(str).isin(bad_ids)
    df.loc[mask, "odds"]       = np.nan
    df.loc[mask, "popularity"] = np.nan

    after_odds_valid = pd.to_numeric(df["odds"], errors="coerce").notna().sum()
    after_pop_valid  = pd.to_numeric(df.get("popularity", pd.Series()), errors="coerce").notna().sum()

    logger.info(
        f"\n置換結果:"
        f"\n  odds      有効数: {before_odds_valid:,} → {after_odds_valid:,}"
        f"  (NaN化: {before_odds_valid - after_odds_valid:,}行)"
        f"\n  popularity有効数: {before_pop_valid:,} → {after_pop_valid:,}"
        f"  (NaN化: {before_pop_valid - after_pop_valid:,}行)"
    )

    # 上書き保存（.gz なら gz で）
    out = GZ_PATH if GZ_PATH.exists() else CSV_PATH
    if str(out).endswith(".gz"):
        df.to_csv(out, index=False, encoding="utf-8-sig", compression="gzip")
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="cleaned_races.csv.gz の汚染 odds/popularity を NaN に置換する"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="実際に上書き保存する（デフォルトは dry-run）",
    )
    args = parser.parse_args()
    purge(apply=args.apply)
