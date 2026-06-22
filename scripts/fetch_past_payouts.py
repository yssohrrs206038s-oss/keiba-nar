"""
過去レースの払戻金をネットけいばから取得し data/payouts/{race_id}.json に保存する。

results_history.csv に記録された全レース（または指定条件）を対象に、
払戻 JSON がまだ存在しないレースだけ取得する。
--backfill フラグを付けると sanrenpuku_payout / return_total が 0 のまま残っている
実買いレースを再計算して results_history.csv を上書きする。

使い方:
    python scripts/fetch_past_payouts.py                     # 全レース (未保存分のみ)
    python scripts/fetch_past_payouts.py --since 2026-06-01  # 特定日以降
    python scripts/fetch_past_payouts.py --race-id 202630060101
    python scripts/fetch_past_payouts.py --backfill          # CSV も更新
    python scripts/fetch_past_payouts.py --force             # 保存済みも再取得
    python scripts/fetch_past_payouts.py --dry-run           # 書き込みなし(確認用)
"""

import argparse
import json
import logging
import re
import sys
import time
import random
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from keiba_predictor.discord_notify import scrape_payouts  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR      = ROOT / "keiba_predictor" / "data"
HISTORY_PATH  = DATA_DIR / "results_history.csv"
PAYOUTS_DIR   = DATA_DIR / "payouts"

# 実買い時の投資単位（1,000円 → 100円ベース配当 × 10 が return_total）
BETS_PER_RACE = 1000


def _payout_str_to_int(s: str) -> int:
    if not s:
        return 0
    try:
        return int(re.sub(r"[¥,\s円]", "", str(s)))
    except ValueError:
        return 0


def _load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        logger.error(f"results_history.csv が見つかりません: {HISTORY_PATH}")
        sys.exit(1)
    return pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)


def _save_history(df: pd.DataFrame) -> None:
    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"results_history.csv を更新しました ({len(df)} 行)")


def _fetch_and_save(race_id: str, session: requests.Session, dry_run: bool) -> dict:
    """nar.netkeiba.com から払戻を取得して data/payouts/{race_id}.json に保存する。"""
    payouts = scrape_payouts(race_id, session)
    if not payouts:
        logger.warning(f"  払戻テーブルなし: {race_id}")
        return {}

    if not dry_run:
        PAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PAYOUTS_DIR / f"{race_id}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payouts, f, ensure_ascii=False, indent=2)
        logger.info(f"  保存: {out.name}  券種={list(payouts.keys())}")
    else:
        logger.info(f"  [dry-run] 取得済み: {race_id}  券種={list(payouts.keys())}")
    return payouts


def _backfill_row(row: pd.Series, payouts: dict) -> dict:
    """
    sanrenpuku_payout=0 の実買い行を再計算する。
    actual1/2/3_num を 三連複コンボとして払戻を検索する。
    Returns: 更新する列の dict（変更なければ空）
    """
    sanren_hit = str(row.get("sanrenpuku_hit", "False")).strip().lower() == "true"
    if not sanren_hit:
        return {}

    try:
        a1 = int(float(row.get("actual1_num") or 0))
        a2 = int(float(row.get("actual2_num") or 0))
        a3 = int(float(row.get("actual3_num") or 0))
    except (ValueError, TypeError):
        return {}
    if 0 in (a1, a2, a3):
        return {}

    combo = "-".join(str(n) for n in sorted([a1, a2, a3]))
    # 三連複/三連単の候補を検索
    payout_val = 0
    for key in ("三連複", "3連複"):
        for entry in payouts.get(key, []):
            entry_combo = str(entry.get("combo", ""))
            entry_nums = sorted(int(x) for x in re.findall(r"\d+", entry_combo))
            if entry_nums == sorted([a1, a2, a3]):
                payout_val = int(entry.get("amount") or 0)
                break
        if payout_val:
            break

    if not payout_val:
        logger.debug(f"  三連複コンボ {combo} が払戻テーブルに見つかりません")
        return {}

    try:
        bet_total = int(float(row.get("bet_total") or BETS_PER_RACE))
    except (ValueError, TypeError):
        bet_total = BETS_PER_RACE
    multiplier = max(bet_total // 100, 1)
    return_total = payout_val * multiplier

    return {
        "sanrenpuku_payout": str(payout_val),
        "return_total":      str(return_total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="過去レース払戻金を取得・保存する")
    parser.add_argument("--since",    metavar="YYYY-MM-DD", help="この日付以降のレースのみ対象")
    parser.add_argument("--race-id",  metavar="RACE_ID",    help="特定の race_id のみ対象")
    parser.add_argument("--backfill", action="store_true",  help="sanrenpuku_payout=0 の行を再計算して CSV 更新")
    parser.add_argument("--force",    action="store_true",  help="既存の払戻 JSON も再取得")
    parser.add_argument("--dry-run",  action="store_true",  help="書き込みを行わない（確認用）")
    args = parser.parse_args()

    df = _load_history()

    # ── 対象レースを絞り込む ──────────────────────────────────
    targets = df.copy()

    if args.race_id:
        targets = targets[targets["race_id"] == args.race_id]
        if targets.empty:
            logger.error(f"race_id {args.race_id} は results_history.csv に存在しません")
            sys.exit(1)

    if args.since:
        targets = targets[targets["date"] >= args.since]

    # 重複排除（同一 race_id の複数行は1回だけ取得）
    race_ids = targets["race_id"].dropna().unique().tolist()
    logger.info(f"対象レース: {len(race_ids)} 件 (--force={args.force}, --backfill={args.backfill})")

    # ── 払戻取得ループ ────────────────────────────────────────
    session = requests.Session()
    fetched: dict[str, dict] = {}   # race_id → payouts

    skipped = 0
    for i, race_id in enumerate(race_ids):
        json_path = PAYOUTS_DIR / f"{race_id}.json"

        if json_path.exists() and not args.force:
            with open(json_path, encoding="utf-8") as f:
                fetched[race_id] = json.load(f)
            skipped += 1
            continue

        logger.info(f"[{i+1}/{len(race_ids)}] 取得中: {race_id}")
        payouts = _fetch_and_save(race_id, session, dry_run=args.dry_run)
        if payouts:
            fetched[race_id] = payouts
        # サーバー負荷軽減
        time.sleep(random.uniform(1.2, 2.2))

    logger.info(f"取得完了: {len(fetched) - skipped} 件新規, {skipped} 件スキップ(既存)")

    # ── バックフィル ──────────────────────────────────────────
    if not args.backfill:
        return

    logger.info("backfill モード: sanrenpuku_payout=0 の実買い行を再計算します")
    updated_count = 0
    df_work = df.copy()

    for idx, row in df_work.iterrows():
        race_id = str(row.get("race_id", ""))
        try:
            bet_total = float(row.get("bet_total") or 0)
        except (ValueError, TypeError):
            bet_total = 0

        sanren_hit = str(row.get("sanrenpuku_hit", "False")).strip().lower() == "true"
        try:
            sanren_payout = float(row.get("sanrenpuku_payout") or 0)
        except (ValueError, TypeError):
            sanren_payout = 0

        # 対象: 実買い(bet_total>0) かつ hit=True かつ payout 未記録
        if not (bet_total > 0 and sanren_hit and sanren_payout == 0):
            continue

        payouts = fetched.get(race_id)
        if payouts is None:
            logger.warning(f"  {race_id}: 払戻データなし → スキップ")
            continue

        updates = _backfill_row(row, payouts)
        if not updates:
            continue

        for col, val in updates.items():
            df_work.at[idx, col] = val
        logger.info(
            f"  更新: {race_id} {row.get('race_name','')} "
            f"sanrenpuku_payout={updates.get('sanrenpuku_payout')} "
            f"return_total={updates.get('return_total')}"
        )
        updated_count += 1

    logger.info(f"backfill: {updated_count} 行更新")

    if updated_count > 0 and not args.dry_run:
        _save_history(df_work)
    elif updated_count > 0:
        logger.info("[dry-run] CSV は更新しません")
    else:
        logger.info("更新対象行なし")


if __name__ == "__main__":
    main()
