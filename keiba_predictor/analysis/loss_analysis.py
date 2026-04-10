"""
NAR成績レポート（日次・週次）

results_history.csv から的中率・回収率を集計し Discord に送信する。

日次レポート（毎日送信）: 当日分の成績
週次レポート（日曜のみ）: 月曜〜日曜の週間成績

ワイド1点1,000円戦略前提で投資・回収を計算する。

使い方:
    python -m keiba_predictor.analysis.loss_analysis            # 日次
    python -m keiba_predictor.analysis.loss_analysis --weekly   # 週次
"""

import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_PATH = DATA_DIR / "results_history.csv"

# 戦略変更日: ワイド1点1,000円固定
STRATEGY_START = "2026-04-07"
BET_PER_RACE = 1000

# JST基準の今日の日付を取得
def _today_jst() -> date:
    return datetime.now(timezone(timedelta(hours=9))).date()


def _load_cache() -> dict:
    """predictions_cache.json を読み込む。"""
    cache_path = HISTORY_PATH.parent / "predictions_cache.json"
    if not cache_path.exists():
        return {}
    try:
        import json
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _calc_return(row: dict, bet_amount: int = 1000) -> int:
    """ワイド100円ベース配当 × (bet_amount/100) = 購入時の払戻"""
    if row.get("wide_hit") != "True":
        return 0
    try:
        return int(row.get("wide_payout") or 0) * (bet_amount // 100)
    except (ValueError, TypeError):
        return 0


def _aggregate(rows: list[dict], cache: dict = None) -> dict:
    """的中率・投資・回収・回収率を集計（ストリーク増額対応）"""
    n = len(rows)
    if n == 0:
        return {"n": 0}

    if cache is None:
        cache = {}

    honmei_hits = sum(1 for r in rows if r.get("fukusho_hit") == "True")
    wide_hits = sum(1 for r in rows if r.get("wide_hit") == "True")

    # ストリーク増額を反映した投資・回収を計算
    bet = 0
    ret = 0
    bet_flat = 0
    ret_flat = 0
    boosted_races = 0
    boosted_venues: dict[str, int] = {}

    for r in rows:
        rid = str(r.get("race_id", ""))
        entry = cache.get(rid, {})
        bs = entry.get("bet_strategy", {})
        cost = bs.get("total_cost", BET_PER_RACE)
        streak = bs.get("streak", 0)

        bet += cost
        ret += _calc_return(r, cost)
        bet_flat += BET_PER_RACE
        ret_flat += _calc_return(r, BET_PER_RACE)

        if streak >= 2:
            boosted_races += 1
            venue = entry.get("venue", "?")
            boosted_venues[venue] = boosted_venues.get(venue, 0) + 1

    profit = ret - bet
    roi = (ret / bet * 100) if bet > 0 else 0
    honmei_rate = honmei_hits / n * 100 if n > 0 else 0
    wide_rate = wide_hits / n * 100 if n > 0 else 0
    boost_effect = (ret - ret_flat) - (bet - bet_flat)  # 増額による純利益差

    return {
        "n": n,
        "honmei_hits": honmei_hits,
        "honmei_rate": honmei_rate,
        "wide_hits": wide_hits,
        "wide_rate": wide_rate,
        "bet": bet,
        "ret": ret,
        "profit": profit,
        "roi": roi,
        "bet_flat": bet_flat,
        "boosted_races": boosted_races,
        "boosted_venues": boosted_venues,
        "boost_effect": boost_effect,
    }


def _load_rows() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)
        return df.to_dict("records")
    except Exception:
        return []


def analyze_daily(target_date: str = None) -> str:
    """日次レポート: 当日分の成績."""
    if target_date is None:
        target_date = _today_jst().isoformat()

    rows = _load_rows()
    daily_rows = [
        r for r in rows
        if str(r.get("date", ""))[:10] == target_date
        and str(r.get("date", ""))[:10] >= STRATEGY_START
    ]

    if not daily_rows:
        return ""

    cache = _load_cache()
    s = _aggregate(daily_rows, cache)
    sep = "━" * 16
    lines = [
        "📊 **【KEIBA EDGE】本日の成績**",
        f"📅 {target_date}",
        sep,
        f"対象レース: {s['n']}戦",
        f"本命的中率: {s['honmei_rate']:.0f}%（{s['honmei_hits']}/{s['n']}）",
        f"ワイド的中率: {s['wide_rate']:.0f}%（{s['wide_hits']}/{s['n']}）",
        f"回収率: {s['roi']:.0f}%",
        sep,
    ]

    # ストリーク増額情報
    if s.get("boosted_races", 0) > 0:
        bv = "・".join(f"{v}{c}" for v, c in sorted(s["boosted_venues"].items()))
        effect = s["boost_effect"]
        sign = "+" if effect >= 0 else ""
        lines.extend([
            f"🔥 増額レース: {s['boosted_races']}戦（{bv}）",
            f"通常投資: {s['bet_flat']:,}円 → 実投資: {s['bet']:,}円",
            f"増額効果: {sign}{effect:,}円",
            sep,
        ])

    return "\n".join(lines)


def analyze_weekly(target_date: str = None) -> str:
    """週次レポート: 月曜〜日曜の集計."""
    if target_date is None:
        target_date = _today_jst()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    week_start = target_date - timedelta(days=target_date.weekday())  # 月曜
    week_end = week_start + timedelta(days=6)  # 日曜
    ws = week_start.isoformat()
    we = week_end.isoformat()

    rows = _load_rows()
    week_rows = [
        r for r in rows
        if ws <= str(r.get("date", ""))[:10] <= we
        and str(r.get("date", ""))[:10] >= STRATEGY_START
    ]

    if not week_rows:
        return ""

    cache = _load_cache()
    s = _aggregate(week_rows, cache)
    sep = "━" * 16
    profit_sign = "+" if s["profit"] >= 0 else ""
    lines = [
        "📊 **【KEIBA EDGE】今週の成績**",
        f"📅 {ws} 〜 {we}",
        sep,
        f"対象レース: {s['n']}戦",
        f"本命的中率: {s['honmei_rate']:.0f}%（{s['honmei_hits']}/{s['n']}）",
        f"ワイド的中率: {s['wide_rate']:.0f}%（{s['wide_hits']}/{s['n']}）",
        f"回収率: {s['roi']:.0f}%",
        sep,
        f"週間投資額: {s['bet']:,}円",
        f"週間回収額: {s['ret']:,}円",
        f"週間損益: {profit_sign}{s['profit']:,}円",
        sep,
    ]

    if s.get("boosted_races", 0) > 0:
        bv = "・".join(f"{v}{c}" for v, c in sorted(s["boosted_venues"].items()))
        effect = s["boost_effect"]
        sign = "+" if effect >= 0 else ""
        lines.extend([
            f"🔥 増額レース: {s['boosted_races']}戦（{bv}）",
            f"増額効果: {sign}{effect:,}円",
            sep,
        ])

    return "\n".join(lines)


# 後方互換
def analyze_week() -> str:
    return analyze_daily()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    weekly = "--weekly" in sys.argv
    report = analyze_weekly() if weekly else analyze_daily()

    if not report:
        print("分析対象のデータがありません")
        return

    print(report)

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

    # X（Twitter）に当日の結果まとめを投稿（日次のみ）
    if not weekly and os.environ.get("ENABLE_X_POST", "false").lower() == "true":
        try:
            from keiba_predictor.x_post import post_daily_result_summary
            today_str = _today_jst().isoformat()
            today_rows = [r for r in _load_rows() if str(r.get("date", "")).startswith(today_str)]
            posted = post_daily_result_summary(today_rows)
            print(f"X結果まとめ投稿: {posted}件")
        except Exception as e:
            print(f"X結果まとめ投稿失敗: {e}")


if __name__ == "__main__":
    main()
