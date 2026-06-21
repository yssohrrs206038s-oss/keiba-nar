"""
scripts/fetch_past_payouts.py

results_history.csv の全race_idについて複勝配当を取得し、
k係数(複勝オッズ ≈ 1 + (単勝オッズ-1)*k)を実測する。

使い方:
  python scripts/fetch_past_payouts.py

出力:
  keiba_predictor/data/payouts/{race_id}.json  (レースごとの払戻データ)
  標準出力に k 係数の集計
"""
import json
import sys
import time
import random
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from keiba_predictor.scraper.netkeiba_scraper import NAR_RESULT_URL, _get
from keiba_predictor.discord_notify import scrape_payouts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent.parent / "keiba_predictor" / "data"
PAYOUT_DIR = DATA_DIR / "payouts"
HISTORY_CSV = DATA_DIR / "results_history.csv"
CLEANED_CSV_GZ = DATA_DIR / "cleaned_races.csv.gz"
CLEANED_CSV    = DATA_DIR / "cleaned_races.csv"

SLEEP_SEC = 2.0  # リクエスト間隔（IPブロック防止）


def load_race_ids() -> list[str]:
    df = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    ids = df["race_id"].dropna().astype(str).unique().tolist()
    logger.info(f"results_history.csv: {len(ids)} レース")
    return ids


def fetch_all_payouts(race_ids: list[str]) -> None:
    PAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; keiba-nar-bot/1.0)"})

    skip = 0
    fetch = 0
    errors = 0

    for i, race_id in enumerate(race_ids, 1):
        out_path = PAYOUT_DIR / f"{race_id}.json"
        if out_path.exists():
            skip += 1
            continue

        try:
            payouts = scrape_payouts(race_id, session)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payouts, f, ensure_ascii=False, indent=2)
            fetch += 1
            logger.info(f"  [{i}/{len(race_ids)}] {race_id} → {list(payouts.keys())}")
        except Exception as e:
            logger.error(f"  [{i}/{len(race_ids)}] {race_id} エラー: {e}")
            errors += 1

        time.sleep(SLEEP_SEC + random.uniform(0, 0.5))

    logger.info(f"完了: 取得={fetch}, スキップ={skip}, エラー={errors}")


def load_odds_data() -> pd.DataFrame:
    """cleaned_races.csv から単勝オッズ・頭数・馬番を読み込む。"""
    path = CLEANED_CSV_GZ if CLEANED_CSV_GZ.exists() else CLEANED_CSV
    logger.info(f"オッズデータ読み込み中: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                     usecols=["race_id", "horse_number", "odds"])
    df["race_id"] = df["race_id"].astype(str)
    df["odds"]    = pd.to_numeric(df["odds"], errors="coerce")
    # 頭数: race_id ごとの馬番の最大値
    df["n_horses"] = df.groupby("race_id")["horse_number"].transform("count")
    return df


def calc_k_stats() -> None:
    """全JSONから複勝配当を集計してk係数を算出する。"""
    odds_df = load_odds_data()

    records = []
    json_files = sorted(PAYOUT_DIR.glob("*.json"))
    logger.info(f"JSON集計: {len(json_files)} ファイル")

    for jf in json_files:
        race_id = jf.stem
        try:
            with open(jf, encoding="utf-8") as f:
                payouts = json.load(f)
        except Exception:
            continue

        if payouts.get("_refunded"):
            continue

        fukusho_list = payouts.get("複勝", [])
        if not fukusho_list:
            continue

        # このレースの単勝オッズ・頭数
        race_odds = odds_df[odds_df["race_id"] == race_id].copy()
        if race_odds.empty:
            continue

        n_horses = race_odds["n_horses"].iloc[0]

        for entry in fukusho_list:
            horse_num_str = str(entry.get("combo", "")).strip()
            fuku_amt      = entry.get("amount")  # 100円あたりの配当(円)
            if not horse_num_str or fuku_amt is None:
                continue

            # 複勝オッズ = 配当金額 / 100
            fuku_odds = fuku_amt / 100.0

            # 対応する単勝オッズを検索
            row = race_odds[race_odds["horse_number"].astype(str) == horse_num_str]
            if row.empty:
                continue
            tan_odds = row["odds"].iloc[0]
            if pd.isna(tan_odds) or tan_odds <= 1.0:
                continue

            k = (fuku_odds - 1.0) / (tan_odds - 1.0)
            if k <= 0 or k > 1.0:
                continue  # 異常値除外

            records.append({
                "race_id":    race_id,
                "horse_num":  horse_num_str,
                "n_horses":   n_horses,
                "tan_odds":   tan_odds,
                "fuku_odds":  fuku_odds,
                "k":          k,
            })

    if not records:
        print("k係数を算出できるデータなし（複勝配当が取得できていない可能性）")
        return

    kdf = pd.DataFrame(records)
    print(f"\n{'='*55}")
    print(f"k係数集計  サンプル数: {len(kdf)}")
    print(f"{'='*55}")
    print(f"  全体  中央値k: {kdf['k'].median():.4f}  平均k: {kdf['k'].mean():.4f}")

    small = kdf[kdf["n_horses"] <= 8]
    large = kdf[kdf["n_horses"] >= 9]
    if len(small) > 0:
        print(f"  8頭以下(n={len(small)})  中央値k: {small['k'].median():.4f}  平均k: {small['k'].mean():.4f}")
    if len(large) > 0:
        print(f"  9頭以上(n={len(large)})  中央値k: {large['k'].median():.4f}  平均k: {large['k'].mean():.4f}")

    # 分布確認
    print(f"\n  k分布:")
    for lo, hi in [(0.0, 0.10), (0.10, 0.15), (0.15, 0.20),
                   (0.20, 0.25), (0.25, 0.30), (0.30, 0.35),
                   (0.35, 0.40), (0.40, 0.50), (0.50, 1.01)]:
        cnt = ((kdf["k"] >= lo) & (kdf["k"] < hi)).sum()
        bar = "█" * (cnt // max(len(kdf) // 50, 1))
        print(f"    {lo:.2f}-{hi:.2f}: {cnt:4d}  {bar}")

    # 単勝オッズ帯別の中央値k
    print(f"\n  単勝オッズ帯別 中央値k:")
    for lo, hi, label in [(1, 3, "~3倍"), (3, 5, "3-5倍"), (5, 10, "5-10倍"),
                           (10, 20, "10-20倍"), (20, 999, "20倍~")]:
        sub = kdf[(kdf["tan_odds"] >= lo) & (kdf["tan_odds"] < hi)]
        if len(sub) == 0:
            continue
        print(f"    {label:<10} n={len(sub):4d}  中央値k={sub['k'].median():.4f}")

    print(f"\n推奨k係数:")
    print(f"  8頭以下: {small['k'].median():.3f}" if len(small) > 0 else "  8頭以下: データなし")
    print(f"  9頭以上: {large['k'].median():.3f}" if len(large) > 0 else "  9頭以上: データなし")


def main():
    race_ids = load_race_ids()
    logger.info(f"払戻データ取得開始 (計{len(race_ids)}レース, 間隔{SLEEP_SEC}秒)")
    fetch_all_payouts(race_ids)
    logger.info("k係数を計算中...")
    calc_k_stats()


if __name__ == "__main__":
    main()
