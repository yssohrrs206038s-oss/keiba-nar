"""
NAR成績レポート（日次・週次）

results_history.csv から的中率・回収率を集計し Discord に送信する。

日次レポート（毎日送信）: 当日分の成績
週次レポート（日曜のみ）: 月曜〜日曜の週間成績

◎オッズで自動切替（◎≤2倍→3連複◎○▲1000円、◎>2倍→ワイド3点900円）。

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

# 戦略変更日
STRATEGY_START = "2026-04-07"
BET_PER_RACE = 1000  # デフォルト（実際はbet_totalカラムから取得）

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


def _calc_return(row: dict, bet_amount: int = None) -> int:
    """results_history.csvのreturn_totalを直接返す。"""
    try:
        return int(float(row.get("return_total") or 0))
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

    bet = 0
    ret = 0

    for r in rows:
        try:
            cost = int(float(r.get("bet_total") or 0))
        except (ValueError, TypeError):
            cost = 0
        bet += cost
        ret += _calc_return(r)

    profit = ret - bet
    roi = (ret / bet * 100) if bet > 0 else 0
    honmei_rate = honmei_hits / n * 100 if n > 0 else 0
    wide_rate = wide_hits / n * 100 if n > 0 else 0

    # 3連複的中数もカウント
    sanren_hits = sum(1 for r in rows if r.get("sanrenpuku_hit") == "True")

    return {
        "n": n,
        "honmei_hits": honmei_hits,
        "honmei_rate": honmei_rate,
        "wide_hits": wide_hits,
        "wide_rate": wide_rate,
        "sanren_hits": sanren_hits,
        "bet": bet,
        "ret": ret,
        "profit": profit,
        "roi": roi,
    }


def _load_rows() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        df = pd.read_csv(HISTORY_PATH, encoding="utf-8-sig", dtype=str)
        return df.to_dict("records")
    except Exception:
        return []


def _rolling_30_section(all_rows: list[dict], cache: dict) -> list[str]:
    """直近30戦パフォーマンスセクションを生成する。"""
    qualified = [
        r for r in all_rows
        if str(r.get("date", ""))[:10] >= STRATEGY_START
        and int(float(r.get("bet_total") or 0)) > 0  # シャドウ(見送り)除外
    ]
    # 日付降順で直近30戦を取得
    qualified.sort(key=lambda r: str(r.get("date", ""))[:10], reverse=True)
    recent = qualified[:30]
    if not recent:
        return []

    s = _aggregate(recent, cache)
    roi = s["roi"]
    sep = "━" * 16

    lines = [
        sep,
        f"📈 **直近30戦パフォーマンス**（{s['n']}戦集計）",
        f"ワイド的中率: {s['wide_rate']:.0f}%（{s['wide_hits']}/{s['n']}）",
        f"回収率: {roi:.0f}%",
        f"損益: {'+' if s['profit'] >= 0 else ''}{s['profit']:,}円",
    ]

    if roi < 120:
        lines.append(f"⚠️ 直近30戦ROI {roi:.0f}% — フィルタ見直し推奨")
    elif roi > 200:
        lines.append(f"✨ 直近30戦ROI {roi:.0f}% — 好調")

    lines.append(sep)
    return lines


def analyze_interim(target_date: str = None) -> str:
    """途中経過: 当日分の簡易サマリー（12/14/16/18/20時用）."""
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
    from datetime import datetime
    now_jst = datetime.now(__import__('zoneinfo').ZoneInfo("Asia/Tokyo")).strftime("%H:%M")

    return (
        f"📊 途中経過（{now_jst}時点）\n"
        f"対象: {s['n']}戦 / ワイド{s['wide_hits']} 3連複{s['sanren_hits']} / "
        f"回収率 {s['roi']:.0f}% / 損益 {'+' if s['profit'] >= 0 else ''}{s['profit']:,}円"
    )


def analyze_daily(target_date: str = None) -> str:
    """日次レポート: 当日分の成績."""
    if target_date is None:
        target_date = _today_jst().isoformat()

    rows = _load_rows()
    daily_rows = [
        r for r in rows
        if str(r.get("date", ""))[:10] == target_date
        and str(r.get("date", ""))[:10] >= STRATEGY_START
        and int(float(r.get("bet_total") or 0)) > 0  # シャドウ(見送り)除外
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
        f"ワイド的中: {s['wide_hits']}件 / 3連複的中: {s['sanren_hits']}件",
        f"投資: {s['bet']:,}円 → 回収: {s['ret']:,}円",
        f"回収率: {s['roi']:.0f}% / 損益: {'+' if s['profit']>=0 else ''}{s['profit']:,}円",
        sep,
    ]

    # 直近30戦パフォーマンス
    rolling = _rolling_30_section(rows, cache)
    if rolling:
        lines.extend(rolling)

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

    # 直近30戦パフォーマンス
    rolling = _rolling_30_section(rows, cache)
    if rolling:
        lines.extend(rolling)

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
    interim = "--interim" in sys.argv
    if weekly:
        report = analyze_weekly()
    elif interim:
        report = analyze_interim()
    else:
        report = analyze_daily()

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
