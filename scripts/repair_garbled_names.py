"""
results_history.csv の文字化け馬名を修復するスクリプト。

化けた行（name系カラムにU+FFFDを含む行）のrace_idで結果ページを再取得し、
馬番→馬名の対応で pred1-3_name / actual1-3_name を引き直す。
的中判定は馬番ベースなので名前カラムのみ更新する。

使い方:
  # dry-run（デフォルト）: 変更内容を表示するのみ
  python scripts/repair_garbled_names.py

  # 実際にCSVを上書き保存
  python scripts/repair_garbled_names.py --apply
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT    = Path(__file__).parent.parent
HISTORY_PATH = REPO_ROOT / "keiba_predictor" / "data" / "results_history.csv"
REPLACEMENT  = "�"

NAME_COLS = [
    "pred1_name", "pred2_name", "pred3_name",
    "actual1_name", "actual2_name", "actual3_name",
]
NUM_COLS = [
    "pred1_num", "pred2_num", "pred3_num",
    "actual1_num", "actual2_num", "actual3_num",
]


def _has_garble(row: pd.Series) -> bool:
    return any(REPLACEMENT in str(row.get(c, "")) for c in NAME_COLS)


def _fetch_horse_map(race_id: str, session: requests.Session) -> dict[int, str]:
    """race_idの結果ページから 馬番→馬名 の dict を返す。"""
    # ローカルの scraper を使う（_get のエンコーディング修正済み）
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from keiba_predictor.scraper.netkeiba_scraper import (
        NAR_RESULT_URL, _get, HEADERS,
    )

    url = f"{NAR_RESULT_URL}?race_id={race_id}"
    soup = _get(url, session, encoding="euc-jp")
    if soup is None:
        logger.warning(f"  ページ取得失敗: {race_id}")
        return {}

    horse_map: dict[int, str] = {}

    # 結果テーブルを探す
    result_table = None
    for sel in [
        "table.RaceTable01", "table.ResultMain", "table.race_table_01",
        "table.nk_tb_common",
    ]:
        result_table = soup.select_one(sel)
        if result_table:
            break
    if result_table is None:
        for tbl in soup.select("table"):
            if "着順" in tbl.get_text() and "馬名" in tbl.get_text():
                result_table = tbl
                break

    if result_table is None:
        logger.warning(f"  テーブル未検出: {race_id}")
        return {}

    # ヘッダから 馬番・馬名 のカラムインデックスを特定
    header_row = result_table.select_one("thead tr") or result_table.select_one("tr")
    raw_headers: list[str] = []
    if header_row:
        raw_headers = [th.get_text(strip=True) for th in header_row.select("th")]
        if not raw_headers:
            raw_headers = [td.get_text(strip=True) for td in header_row.select("td")]

    num_idx  = next((i for i, h in enumerate(raw_headers) if "馬番" in h), 2)
    name_idx = next((i for i, h in enumerate(raw_headers) if "馬名" in h), 3)

    for tr in result_table.select("tr")[1:]:
        tds = tr.select("td")
        if len(tds) <= max(num_idx, name_idx):
            continue
        try:
            horse_num = int(tds[num_idx].get_text(strip=True))
        except (ValueError, IndexError):
            continue
        horse_el = tds[name_idx].select_one("a")
        horse_name = horse_el.get_text(strip=True) if horse_el else tds[name_idx].get_text(strip=True)
        if horse_name:
            horse_map[horse_num] = horse_name

    return horse_map


def repair(apply: bool) -> None:
    if not HISTORY_PATH.exists():
        logger.error(f"ファイルが見つかりません: {HISTORY_PATH}")
        return

    df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)
    garbled_mask = df.apply(_has_garble, axis=1)
    garbled_rows = df[garbled_mask]

    if garbled_rows.empty:
        logger.info("文字化け行はありませんでした。")
        return

    logger.info(f"文字化け行: {len(garbled_rows)} 件")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/136.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    })

    changes: list[tuple[int, str, str, str]] = []  # (idx, col, old, new)

    for idx, row in garbled_rows.iterrows():
        race_id = str(row["race_id"])
        logger.info(f"処理中: race_id={race_id}")

        horse_map = _fetch_horse_map(race_id, session)
        if not horse_map:
            logger.warning(f"  horse_map 取得失敗、スキップ: {race_id}")
            time.sleep(2)
            continue

        for name_col, num_col in zip(NAME_COLS, NUM_COLS):
            old_name = str(row.get(name_col, ""))
            if REPLACEMENT not in old_name:
                continue
            try:
                horse_num = int(str(row.get(num_col, "")))
            except (ValueError, TypeError):
                logger.warning(f"  {num_col} が無効: {row.get(num_col)!r}")
                continue
            new_name = horse_map.get(horse_num)
            if new_name is None:
                logger.warning(f"  馬番{horse_num} が horse_map に未存在: {race_id}")
                continue
            changes.append((idx, name_col, old_name, new_name))
            logger.info(f"  {name_col}[{horse_num}番]: {old_name!r} → {new_name!r}")
            if apply:
                df.at[idx, name_col] = new_name

        time.sleep(2)

    logger.info(f"\n修正対象: {len(changes)} カラム（行{len(set(c[0] for c in changes))}件）")

    if not apply:
        logger.info("dry-run モードのため保存しません。--apply で実際に保存されます。")
        return

    df.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"保存完了: {HISTORY_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="results_history.csv の文字化け馬名を修復する")
    parser.add_argument(
        "--apply", action="store_true",
        help="指定するとCSVを実際に上書き保存する（デフォルトはdry-run）",
    )
    args = parser.parse_args()
    repair(apply=args.apply)
