"""
X（Twitter）自動投稿モジュール

【環境変数】
    TWITTER_API_KEY              : Consumer Key（API Key）
    TWITTER_API_SECRET           : Consumer Secret（API Secret）
    TWITTER_ACCESS_TOKEN         : Access Token
    TWITTER_ACCESS_TOKEN_SECRET  : Access Token Secret

環境変数が未設定の場合は投稿をスキップし、エラーにはなりません。
"""

import logging
import os
import re

import pandas as pd

logger = logging.getLogger(__name__)

# 文字数の安全上限
_CHAR_LIMIT = 140

# グレード判定パターン
_GRADE_PATS = [
    ("GI",   re.compile(r"[（(]G\s*[1Ⅰ][）)]|[（(]GI[）)]",   re.I)),
    ("GII",  re.compile(r"[（(]G\s*[2Ⅱ][）)]|[（(]GII[）)]",  re.I)),
    ("GIII", re.compile(r"[（(]G\s*[3Ⅲ][）)]|[（(]GIII[）)]", re.I)),
]


# ── 内部ユーティリティ ────────────────────────────────────────────────

def _build_client():
    """tweepy.Client を環境変数から構築する。未設定なら None を返す。"""
    try:
        import tweepy
    except ImportError:
        logger.warning("[X] tweepy がインストールされていません: pip install tweepy")
        return None

    keys = {
        "consumer_key":        os.environ.get("TWITTER_API_KEY", ""),
        "consumer_secret":     os.environ.get("TWITTER_API_SECRET", ""),
        "access_token":        os.environ.get("TWITTER_ACCESS_TOKEN", ""),
        "access_token_secret": os.environ.get("TWITTER_ACCESS_TOKEN_SECRET", ""),
    }
    if not all(keys.values()):
        missing = [k for k, v in keys.items() if not v]
        logger.info(f"[X] 資格情報未設定のためスキップ ({missing})")
        return None
    return tweepy.Client(**keys)


def _grade_label(race_name: str) -> str:
    for label, pat in _GRADE_PATS:
        if pat.search(race_name):
            return label
    return ""


def _short_name(race_name: str) -> str:
    """括弧内グレード表記を除いた短縮レース名。ハッシュタグ用。"""
    return re.sub(r"[（(]G[^）)]*[）)]", "", race_name).strip()


def _ev_stars(ev: float) -> str:
    if ev >= 15:
        return "★★★"
    elif ev >= 12:
        return "★★"
    elif ev >= 9:
        return "★"
    return ""


def _safe_post(client, text: str) -> bool:
    """ツイートを投稿し、成否を返す。上限超は末尾を切り詰める。"""
    if len(text) > _CHAR_LIMIT:
        text = text[: _CHAR_LIMIT - 1] + "…"
    try:
        resp = client.create_tweet(text=text)
        tweet_id = resp.data.get("id", "?")
        logger.info(f"[X] 投稿完了 id={tweet_id}")
        return True
    except Exception as e:
        logger.warning(f"[X] 投稿失敗: {e}")
        return False


# ── NAR予想ツイート ─────────────────────────────────────────────────────
# ※ keiba-nar リポジトリ専用。Twitter API キーは keiba-predictor と同一の
#    Secret を keiba-nar リポジトリの GitHub Secrets にも登録すること。

def build_predict_tweet(race_name: str, cache_entry: dict) -> str:
    """旧API互換: 1レース分の予想ツイート（非推奨、会場まとめ版を推奨）。"""
    short = _short_name(race_name)
    venue = cache_entry.get("venue", "")
    tag = f"#地方競馬 #KEIBA_EDGE"
    lines = [f"🏇{venue}{short} AI予想"]
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = cache_entry.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        lines.append(f"{mark}{p['horse_number']}番{p['horse_name']}")
    text = "\n".join(lines) + "\n" + tag
    if len(text) > _CHAR_LIMIT:
        lines = [l for l in lines if not l.startswith("▲")]
        text = "\n".join(lines) + "\n" + tag
    return text


def build_venue_summary_tweet(venue: str, race_entries: list[tuple[str, dict]]) -> str:
    """会場まとめツイートを構築する（140字以内）。

    買い目があるレース（bet_strategy.total_points > 0）のみ表示。
    オッズフィルタで見送りになったレースは除外。

    Args:
        venue: 開催場名（例: "大井"）
        race_entries: [(race_id, cache_entry), ...] 同じ会場の全レース

    Returns:
        ツイート本文。買い目0件なら空文字列。
    """
    from datetime import date
    today = date.today().strftime("%-m/%-d") if hasattr(date.today(), "strftime") else date.today().isoformat()
    try:
        today = date.today().strftime("%-m/%-d")
    except ValueError:
        today = date.today().strftime("%#m/%#d")  # Windows

    tag = f"#KEIBA_EDGE #{venue}競馬"
    header = f"🏇{venue} {today} AI注目買い目"

    # 買い目があるレースだけ抽出（race_idでソート）
    valid: list[tuple[int, str]] = []  # (race_num, line)
    for rid, entry in sorted(race_entries, key=lambda x: x[0]):
        bs = entry.get("bet_strategy", {})
        if not bs or bs.get("total_points", 0) == 0:
            continue  # 見送り or 買い目なしはスキップ
        wide = bs.get("wide", [])
        if not wide:
            continue
        w = wide[0]
        # レース番号
        try:
            race_num = int(rid[10:12])
        except (ValueError, IndexError):
            continue
        race_name = entry.get("race_name", "")
        # 重賞・特別レース判定
        is_special = any(kw in race_name for kw in ("重賞", "特別", "杯", "賞", "ステークス", "(G"))
        suffix = "(重賞)" if "重賞" in race_name else ("★" if is_special else "")
        line = f"{race_num}R ◎{w['nums'][0]}-○{w['nums'][1]}{suffix}"
        valid.append((race_num, line))

    if not valid:
        return ""

    lines = [header]
    for _, line in valid:
        test = "\n".join(lines + [line]) + "\n" + tag
        if len(test) <= _CHAR_LIMIT:
            lines.append(line)
        else:
            break

    return "\n".join(lines) + "\n" + tag


def post_predict_tweet(race_name: str, cache_entry: dict) -> bool:
    """旧API互換: 1レース分の予想ツイートを投稿（非推奨）。"""
    client = _build_client()
    if client is None:
        return False
    text = build_predict_tweet(race_name, cache_entry)
    print(f"[X予想ツイート]\n{text}", flush=True)
    return _safe_post(client, text)


def post_venue_summary_tweets(cache: dict) -> int:
    """会場ごとにまとめツイートを投稿する。

    Args:
        cache: predictions_cache.json 全体

    Returns:
        投稿成功件数
    """
    client = _build_client()
    if client is None:
        return 0

    # 会場別にレースをグループ化
    venue_groups: dict[str, list[tuple[str, dict]]] = {}
    for rid, entry in cache.items():
        if rid.startswith("_") or not isinstance(entry, dict):
            continue
        venue = entry.get("venue", "")
        if not venue:
            continue
        venue_groups.setdefault(venue, []).append((rid, entry))

    posted = 0
    for venue, entries in sorted(venue_groups.items()):
        text = build_venue_summary_tweet(venue, entries)
        if not text:
            logger.info(f"  [X] {venue}: 買い目なしスキップ")
            continue
        print(f"[X会場まとめ {venue}]\n{text}", flush=True)
        if _safe_post(client, text):
            posted += 1
            import time
            time.sleep(2)  # API制限対策
    return posted


# ── 結果ツイート ──────────────────────────────────────────────────────

def build_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
    roi_pct: float,
) -> str:
    from keiba_predictor.discord_notify import (
        _check_sanrenpuku_raw, _check_umaren_raw,
    )

    grade = _grade_label(race_name)
    short = _short_name(race_name)

    predicted_nums = pred.get("predicted_top3_nums", [])
    ana_horse_num = pred.get("ana_horse_num")

    # 本命情報
    honmei = pred.get("honmei", {})
    honmei_num = honmei.get("horse_number")
    honmei_name = honmei.get("horse_name", "")

    # 実際の3着以内
    df = actual_df.copy()
    df["_fp"] = pd.to_numeric(df["finish_position"], errors="coerce")
    top3 = df[df["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)
    actual_nums: list[int] = []
    for _, r in top3.iterrows():
        num = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        actual_nums.append(num)

    # 的中判定
    fukusho_hit = honmei_num is not None and int(honmei_num) in actual_nums
    umaren_hit, umaren_pay = _check_umaren_raw(predicted_nums, actual_nums, payouts)
    sanren_hit, sanren_pay = _check_sanrenpuku_raw(
        predicted_nums, actual_nums, payouts, ana_horse_num)

    f_icon = "✅" if fukusho_hit else "❌"
    u_icon = "✅" if umaren_hit else "❌"
    s_icon = "✅" if sanren_hit else "❌"

    SEP = "━━━━━━━━━━━━━━━━"

    any_hit = fukusho_hit or umaren_hit or sanren_hit

    tag = "#地方競馬 #KEIBA_EDGE"

    if sanren_hit:
        pay_str = re.sub(r"[¥,]", "", str(sanren_pay)) if sanren_pay else ""
        lines = [
            f"🎯3連複的中！{short}",
            f"{pay_str}円 回収率{roi_pct:.0f}%" if pay_str and roi_pct > 0
                else (f"{pay_str}円的中！" if pay_str else ""),
            tag,
        ]
    elif any_hit:
        lines = [
            f"🎯的中！{short}",
            f"複勝{f_icon} 馬連{u_icon}",
            tag,
        ]
    else:
        result_line = " ".join(f"{i+1}着{n}番" for i, n in enumerate(actual_nums[:3]))
        lines = [
            f"{short} 結果",
            result_line,
            f"複勝❌馬連❌3連複❌",
        ]

    return "\n".join(line for line in lines if line)


def post_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
) -> bool:
    """結果ツイートを X に投稿する。累計回収率は results_history.csv から自動取得。"""
    client = _build_client()
    if client is None:
        return False

    roi_pct = 0.0
    try:
        from keiba_predictor.history import cumulative_summary, load_history
        roi_pct = cumulative_summary(load_history())["roi"] * 100
    except Exception as e:
        logger.debug(f"[X] 累計回収率取得失敗: {e}")

    text = build_result_tweet(race_name, actual_df, pred, payouts, roi_pct)
    print(f"[X結果ツイート]\n{text}", flush=True)
    return _safe_post(client, text)


# ── 週次サマリーツイート ─────────────────────────────────────────────────

def build_weekly_summary_tweet(results: list[dict]) -> str:
    """週次サマリーツイートを構築する（140字以内）。"""
    total = len(results)
    if not total:
        return ""

    hit_count = sum(1 for r in results if r.get("fukusho") or r.get("umaren") or r.get("sanren"))
    fukusho_hits = sum(1 for r in results if r.get("fukusho"))
    fukusho_rate = (fukusho_hits / total * 100) if total else 0
    total_bet = sum(r.get("bet", 0) for r in results)
    total_ret = sum(r.get("return_total", 0) for r in results)
    roi = (total_ret / total_bet * 100) if total_bet > 0 else 0

    tag = "#地方競馬 #KEIBA_EDGE"
    lines = [
        f"📊地方AI成績 {hit_count}/{total}的中",
        f"複勝{fukusho_rate:.0f}% 回収率{roi:.0f}%",
    ]

    # レース別（余裕がある分だけ）
    race_lines = []
    for r in results:
        name = _short_name(r.get("race_name", ""))[:6]
        f = "○" if r.get("fukusho") else "×"
        u = "○" if r.get("umaren") else "×"
        s = "○" if r.get("sanren") else "×"
        race_lines.append(f"{name}{f}{u}{s}")

    base = "\n".join(lines)
    for rl in race_lines:
        test = base + "\n" + rl + "\n" + tag
        if len(test) <= _CHAR_LIMIT:
            base = base + "\n" + rl
        else:
            break

    return base + "\n" + tag


def post_weekly_summary_tweet(results: list[dict]) -> bool:
    """週次サマリーをXに投稿する。"""
    client = _build_client()
    if client is None:
        return False

    text = build_weekly_summary_tweet(results)
    print(f"[X週次サマリーツイート]\n{text}", flush=True)
    return _safe_post(client, text)
