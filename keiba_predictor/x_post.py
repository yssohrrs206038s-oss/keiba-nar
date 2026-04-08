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

# 文字数の安全上限（X 標準は280字）
_CHAR_LIMIT = 280

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


# ── 日次まとめ投稿（買い目 / 結果） ───────────────────────────────────

def _today_md() -> str:
    from datetime import date
    try:
        return date.today().strftime("%-m/%-d")
    except ValueError:
        return date.today().strftime("%#m/%#d")  # Windows


def _post_thread(client, texts: list[str]) -> int:
    """ツイートをスレッドとして連続投稿する。成功本数を返す。"""
    posted = 0
    reply_to = None
    import time
    for t in texts:
        if len(t) > _CHAR_LIMIT:
            t = t[: _CHAR_LIMIT - 1] + "…"
        try:
            kwargs = {"text": t}
            if reply_to:
                kwargs["in_reply_to_tweet_id"] = reply_to
            resp = client.create_tweet(**kwargs)
            tid = resp.data.get("id")
            logger.info(f"[X] スレッド投稿 id={tid}")
            reply_to = tid
            posted += 1
            time.sleep(2)
        except Exception as e:
            logger.warning(f"[X] スレッド投稿失敗: {e}")
            break
    return posted


def build_daily_bet_summary_tweets(cache: dict) -> list[str]:
    """全会場の買い目を1〜複数ツイートにまとめる（280字超は会場ごとに分割）。

    フォーマット:
        🏇 KEIBA EDGE 本日の買い目 4/8

        🏟 名古屋
        1R ◎4-○9
        ...

        #KEIBA_EDGE #地方競馬
    """
    today = _today_md()
    header = f"🏇 KEIBA EDGE 本日の買い目 {today}"
    tag = "#KEIBA_EDGE #地方競馬"

    # 会場別にまとめる
    venue_groups: dict[str, list[tuple[int, str]]] = {}
    for rid, entry in cache.items():
        if rid.startswith("_") or not isinstance(entry, dict):
            continue
        venue = entry.get("venue", "")
        if not venue:
            continue
        bs = entry.get("bet_strategy", {})
        if not bs or bs.get("total_points", 0) == 0:
            continue
        wide = bs.get("wide", [])
        if not wide:
            continue
        w = wide[0]
        try:
            race_num = int(rid[10:12])
        except (ValueError, IndexError):
            continue
        line = f"{race_num}R ◎{w['nums'][0]}-○{w['nums'][1]}"
        venue_groups.setdefault(venue, []).append((race_num, line))

    if not venue_groups:
        return []

    # 会場ブロックを構築
    venue_blocks: list[str] = []
    for venue in sorted(venue_groups.keys()):
        races = sorted(venue_groups[venue], key=lambda x: x[0])
        block = f"🏟 {venue}\n" + "\n".join(line for _, line in races)
        venue_blocks.append(block)

    # 1ツイートに収まるか試す
    full = header + "\n\n" + "\n\n".join(venue_blocks) + "\n\n" + tag
    if len(full) <= _CHAR_LIMIT:
        return [full]

    # 収まらない場合: 会場ごと（または複数会場）で分割スレッド
    tweets: list[str] = []
    current_parts: list[str] = [header]
    current_len = len(header)
    for block in venue_blocks:
        added_len = len(block) + 2  # "\n\n" 区切り
        if current_len + added_len + 2 + len(tag) > _CHAR_LIMIT and len(current_parts) > 1:
            # 現在のツイートを確定
            tweets.append("\n\n".join(current_parts))
            current_parts = [block]
            current_len = len(block)
        else:
            current_parts.append(block)
            current_len += added_len

    if current_parts:
        # 最終ツイートにタグを付ける
        current_parts.append(tag)
        tweets.append("\n\n".join(current_parts))
    else:
        tweets[-1] = tweets[-1] + "\n\n" + tag

    # 単一会場でも 280 を超える場合に備えてレース行レベルで再分割
    final: list[str] = []
    for tw in tweets:
        if len(tw) <= _CHAR_LIMIT:
            final.append(tw)
            continue
        # 行ごとに分割して詰め直し
        lines = tw.split("\n")
        buf: list[str] = []
        cur = 0
        for ln in lines:
            add = len(ln) + (1 if buf else 0)
            if cur + add > _CHAR_LIMIT and buf:
                final.append("\n".join(buf))
                buf = [ln]
                cur = len(ln)
            else:
                buf.append(ln)
                cur += add
        if buf:
            final.append("\n".join(buf))
    return final


def post_daily_bet_summary(cache: dict) -> int:
    """全会場の買い目まとめをスレッド投稿する。投稿成功本数を返す。"""
    client = _build_client()
    if client is None:
        return 0
    tweets = build_daily_bet_summary_tweets(cache)
    if not tweets:
        logger.info("[X] 買い目まとめ: 対象なしスキップ")
        return 0
    for i, t in enumerate(tweets, 1):
        print(f"[X買い目まとめ {i}/{len(tweets)}]\n{t}", flush=True)
    return _post_thread(client, tweets)


def build_daily_result_summary_tweets(history_rows: list[dict]) -> list[str]:
    """全レースの当落まとめを1〜複数ツイートにまとめる。

    history_rows: results_history.csv の今日分の行（dictリスト）

    フォーマット:
        📊 KEIBA EDGE 本日の結果 4/8

        ✅的中 3R 名古屋 ◎4-○9 1,300円
        ❌不的中 5R 名古屋
        🚫見送り 2R 園田
        ...

        対象: 15戦 / 的中: 5戦 / 回収率: 148%

        #KEIBA_EDGE #地方競馬
    """
    if not history_rows:
        return []

    today = _today_md()
    header = f"📊 KEIBA EDGE 本日の結果 {today}"
    tag = "#KEIBA_EDGE #地方競馬"

    # 各行のサマリ行を構築
    NAR_VENUE_MAP = {
        "30": "門別", "35": "盛岡", "36": "水沢",
        "42": "大井", "43": "船橋", "44": "浦和", "45": "川崎",
        "46": "門別", "47": "名古屋", "48": "園田",
        "50": "笠松", "51": "金沢", "54": "高知", "55": "佐賀", "65": "帯広",
    }

    summary_lines: list[str] = []
    total = 0
    hits = 0
    bet_sum = 0
    ret_sum = 0

    for r in sorted(history_rows, key=lambda x: x.get("race_id", "")):
        rid = str(r.get("race_id", ""))
        if len(rid) < 12:
            continue
        venue = NAR_VENUE_MAP.get(rid[4:6], "")
        try:
            race_num = int(rid[10:12])
        except ValueError:
            continue

        bet = int(float(r.get("bet_total", 0) or 0))
        ret = int(float(r.get("return_total", 0) or 0))
        # 見送り = bet=0
        if bet == 0:
            line = f"🚫見送り {race_num}R {venue}"
            summary_lines.append(line)
            total += 1
            continue

        total += 1
        bet_sum += bet
        wide_hit = str(r.get("wide_hit", "")).lower() == "true"
        # ◎○ 馬番（pred1_num/pred2_num）
        p1 = r.get("pred1_num", "")
        p2 = r.get("pred2_num", "")
        marks = f"◎{p1}-○{p2}" if p1 and p2 else ""
        if wide_hit:
            hits += 1
            ret_sum += ret
            payout = ret  # ret_total はワイド配当(円)
            line = f"✅的中 {race_num}R {venue} {marks} {payout:,}円".rstrip()
        else:
            line = f"❌不的中 {race_num}R {venue} {marks}".rstrip()
        summary_lines.append(line)

    if total == 0:
        return []

    roi = (ret_sum / bet_sum * 100) if bet_sum > 0 else 0
    footer = f"対象: {total}戦 / 的中: {hits}戦 / 回収率: {roi:.0f}%"

    # 1ツイートで収まるか
    full = header + "\n\n" + "\n".join(summary_lines) + "\n\n" + footer + "\n\n" + tag
    if len(full) <= _CHAR_LIMIT:
        return [full]

    # 分割: 1本目に header、最後にfooter+tag、中間は本文行
    tweets: list[str] = []
    # 1本目を header だけで開始
    cur_parts: list[str] = [header, ""]
    cur_len = len(header) + 1
    for ln in summary_lines:
        add = len(ln) + 1
        if cur_len + add > _CHAR_LIMIT and len(cur_parts) > 2:
            tweets.append("\n".join(cur_parts).rstrip())
            cur_parts = [ln]
            cur_len = len(ln)
        else:
            cur_parts.append(ln)
            cur_len += add
    if cur_parts:
        tweets.append("\n".join(cur_parts).rstrip())

    # footer + tag を追加（最後のツイートに余裕があれば結合、なければ別ツイート）
    suffix = "\n\n" + footer + "\n\n" + tag
    if len(tweets[-1]) + len(suffix) <= _CHAR_LIMIT:
        tweets[-1] = tweets[-1] + suffix
    else:
        tweets.append(footer + "\n\n" + tag)
    return tweets


def post_daily_result_summary(history_rows: list[dict]) -> int:
    """全レース結果まとめをスレッド投稿する。投稿成功本数を返す。"""
    client = _build_client()
    if client is None:
        return 0
    tweets = build_daily_result_summary_tweets(history_rows)
    if not tweets:
        logger.info("[X] 結果まとめ: 対象なしスキップ")
        return 0
    for i, t in enumerate(tweets, 1):
        print(f"[X結果まとめ {i}/{len(tweets)}]\n{t}", flush=True)
    return _post_thread(client, tweets)


# ── 結果ツイート ──────────────────────────────────────────────────────

def build_result_tweet(
    race_name: str,
    actual_df: pd.DataFrame,
    pred: dict,
    payouts: dict,
    roi_pct: float,
) -> str:
    """ワイド1点1,000円戦略の結果ツイート."""
    short = _short_name(race_name)

    # 実際の3着以内
    df = actual_df.copy()
    df["_fp"] = pd.to_numeric(df["finish_position"], errors="coerce")
    top3 = df[df["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)
    actual_nums: list[int] = []
    for _, r in top3.iterrows():
        num = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        actual_nums.append(num)

    # ワイド的中判定（bet_strategy.wide）
    bs = pred.get("bet_strategy", {})
    wide_hit = False
    wide_pay = ""
    actual_set = set(actual_nums[:3]) if len(actual_nums) >= 3 else set()
    if bs.get("wide"):
        for w in bs["wide"]:
            a, b = w["nums"]
            if a in actual_set and b in actual_set:
                from keiba_predictor.discord_notify import _get_payout
                wide_pay = _get_payout(payouts, "ワイド", f"{a}-{b}")
                wide_hit = True
                break

    tag = "#地方競馬 #KEIBA_EDGE"

    if wide_hit:
        # 100円ベース配当 × 10 = 1,000円購入時の払戻
        pay_int = 0
        if wide_pay:
            try:
                pay_int = int(re.sub(r"[¥¥,円\s]", "", str(wide_pay))) * 10
            except ValueError:
                pass
        lines = [
            f"🎯ワイド的中！{short}",
            f"{pay_int:,}円 回収率{roi_pct:.0f}%" if pay_int and roi_pct > 0
                else (f"{pay_int:,}円的中！" if pay_int else "ワイド的中！"),
            tag,
        ]
    else:
        result_line = " ".join(f"{i+1}着{n}番" for i, n in enumerate(actual_nums[:3]))
        lines = [
            f"{short} 結果",
            result_line,
            "ワイド❌",
            tag,
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
    """週次サマリーツイートを構築する（140字以内）。

    現戦略: ワイド1点1,000円
    bet=1,000円固定、ret=wide_payout×10で計算
    """
    BET = 1000
    total = len(results)
    if not total:
        return ""

    wide_hits = sum(1 for r in results if r.get("wide"))
    wide_rate = (wide_hits / total * 100) if total else 0
    total_bet = total * BET
    total_ret = sum((int(r.get("wide_payout", 0)) * 10) for r in results if r.get("wide"))
    roi = (total_ret / total_bet * 100) if total_bet > 0 else 0
    profit = total_ret - total_bet
    profit_sign = "+" if profit >= 0 else ""

    tag = "#地方競馬 #KEIBA_EDGE"
    lines = [
        f"📊地方AI成績 {wide_hits}/{total}的中",
        f"ワイド的中率{wide_rate:.0f}% 回収率{roi:.0f}%",
        f"損益 {profit_sign}{profit:,}円",
    ]

    return "\n".join(lines) + "\n" + tag


def post_weekly_summary_tweet(results: list[dict]) -> bool:
    """週次サマリーをXに投稿する。"""
    client = _build_client()
    if client is None:
        return False

    text = build_weekly_summary_tweet(results)
    print(f"[X週次サマリーツイート]\n{text}", flush=True)
    return _safe_post(client, text)
