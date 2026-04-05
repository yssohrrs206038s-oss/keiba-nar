"""
週次外れ分析レポート

results_history.csv と predictions_cache.json / manual_results.json から
外れパターンを分類し Discord に送信する。

使い方:
    python -m keiba_predictor.analysis.loss_analysis
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_PATH = DATA_DIR / "predictions_cache.json"
MANUAL_PATH = DATA_DIR / "manual_results.json"
HISTORY_PATH = DATA_DIR / "results_history.csv"


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def analyze_week() -> str:
    """今週のレース結果を集計し、的中率・回収率のみ返す。"""
    if not HISTORY_PATH.exists():
        return ""

    try:
        hist = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)
    except Exception:
        return ""

    if hist.empty:
        return ""

    n = len(hist)
    f_hits = sum(1 for _, r in hist.iterrows() if r.get("fukusho_hit") == "True")
    u_hits = sum(1 for _, r in hist.iterrows()
                 if r.get("umaren_hit") == "True" or r.get("wide_hit") == "True")
    s_hits = sum(1 for _, r in hist.iterrows() if r.get("sanrenpuku_hit") == "True")

    total_bet = sum(int(r.get("bet_total") or 0) for _, r in hist.iterrows())
    total_ret = sum(int(r.get("return_total") or 0) for _, r in hist.iterrows())
    roi = (total_ret / total_bet * 100) if total_bet > 0 else 0

    sep = "━" * 16
    lines = [
        "📊 **【KEIBA EDGE】今週の成績**",
        sep,
        f"複勝的中率: {f_hits/n*100:.0f}%（{f_hits}/{n}）",
        f"馬連的中率: {u_hits/n*100:.0f}%（{u_hits}/{n}）",
        f"3連複的中率: {s_hits/n*100:.0f}%（{s_hits}/{n}）",
        f"回収率: {roi:.0f}%",
        sep,
    ]

    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    report = analyze_week()
    if not report:
        print("分析対象のデータがありません")
        return

    print(report)

    # Discord送信
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if webhook_url:
        import requests
        try:
            resp = requests.post(webhook_url, json={"content": report}, timeout=15)
            ok = resp.status_code in (200, 204)
            print(f"Discord送信: {'成功' if ok else f'失敗({resp.status_code})'}")
        except Exception as e:
            print(f"Discord送信失敗: {e}")
    else:
        print("DISCORD_WEBHOOK_URL 未設定 → Discord送信スキップ")


if __name__ == "__main__":
    main()
