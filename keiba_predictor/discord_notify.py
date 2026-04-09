"""
週末重賞 自動予想・結果通知 → Discord

【機能1】毎週金曜 09:00 ── 週末重賞の予想を送信
    python -m keiba_predictor.main notify --mode predict

【機能2】毎週日曜 17:00 ── 重賞レースの結果・的中判定を送信
    python -m keiba_predictor.main notify --mode result

【環境変数】
    DISCORD_WEBHOOK_URL : Discord Incoming Webhook URL

【前提条件】
    学習済みモデル: keiba_predictor/model/xgb_model.pkl
    予想はpredict_live()で出馬表を直接スクレイピングするため
    featured_races.csvは不要（キャッシュ優先運用）
"""

import json
import logging
import os
import random
import re
import time
from datetime import date, datetime, timedelta, timezone


def _today_jst() -> date:
    """JST 基準の今日の日付を返す（Actions の UTC ランナー対策）。"""
    return (datetime.now(timezone.utc) + timedelta(hours=9)).date()
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from keiba_predictor.scraper.netkeiba_scraper import (
    _get, _sleep, RACE_RESULT_URL, NAR_RESULT_URL,
)
from keiba_predictor.model.predict import load_model, predict_race, calc_ev_and_flags, format_prediction, _build_course_info
from keiba_predictor.ai_comment import generate_comments

logger = logging.getLogger(__name__)

# ── パス定数 ────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model" / "xgb_model.pkl"
PRED_CACHE = DATA_DIR / "predictions_cache.json"   # 予想キャッシュ
MANUAL_RESULTS = DATA_DIR / "manual_results.json"  # 手動結果入力

# 重賞判定 (G1/G2/G3 を含む括弧表記)
GRADE_RE = re.compile(r"\(G[Ⅰ-Ⅲ1-3]\)|\(GI{1,3}\)")

MARK = {"honmei": "◎", "taikou": "○", "ana": "△", "hoshi": "☆"}

# ── 開催場別 Discord Webhook マップ ─────────────────────────────
# GitHub Secrets に各場の Webhook URL を登録:
#   DISCORD_NAR_OI_WEBHOOK_URL        → 大井チャンネル
#   DISCORD_NAR_FUNABASHI_WEBHOOK_URL → 船橋チャンネル
#   DISCORD_NAR_KAWASAKI_WEBHOOK_URL  → 川崎チャンネル
#   DISCORD_NAR_URAWA_WEBHOOK_URL     → 浦和チャンネル
#   DISCORD_NAR_MONBETSU_WEBHOOK_URL  → 門別チャンネル
#   DISCORD_NAR_NAGOYA_WEBHOOK_URL    → 名古屋チャンネル
#   DISCORD_NAR_KOCHI_WEBHOOK_URL     → 高知チャンネル
#   DISCORD_NAR_SAGA_WEBHOOK_URL      → 佐賀チャンネル
#   DISCORD_WEBHOOK_URL               → その他NAR（デフォルト）
NAR_VENUE_WEBHOOK_MAP: dict[str, str] = {
    "大井":   "DISCORD_NAR_OI_WEBHOOK_URL",
    "船橋":   "DISCORD_NAR_FUNABASHI_WEBHOOK_URL",
    "川崎":   "DISCORD_NAR_KAWASAKI_WEBHOOK_URL",
    "浦和":   "DISCORD_NAR_URAWA_WEBHOOK_URL",
    "門別":   "DISCORD_NAR_MONBETSU_WEBHOOK_URL",
    "名古屋": "DISCORD_NAR_NAGOYA_WEBHOOK_URL",
    "高知":   "DISCORD_NAR_KOCHI_WEBHOOK_URL",
    "佐賀":   "DISCORD_NAR_SAGA_WEBHOOK_URL",
}

# 楽天競馬 場コード（netkeiba race_id[4:6] → 楽天場コード2桁）
RAKUTEN_VENUE_CODE: dict[str, str] = {
    "30": "36",  # 門別
    "35": "10",  # 盛岡
    "36": "11",  # 水沢
    "42": "18",  # 浦和
    "43": "19",  # 船橋
    "44": "20",  # 大井
    "45": "21",  # 川崎
    "46": "22",  # 金沢
    "47": "23",  # 笠松
    "48": "24",  # 名古屋
    "50": "27",  # 園田
    "51": "28",  # 姫路
    "54": "31",  # 高知
    "55": "32",  # 佐賀
}


def _rakuten_race_url(race_id: str, race_date: str = "") -> str:
    """netkeiba race_id から楽天競馬のレースURLを生成する。

    netkeiba race_id: YYYY + 場(2) + MM + DD + RR (12桁)
    楽天URL: https://bet.keiba.rakuten.co.jp/sp/normal/shikibetu/RACEID/{18桁ID}
    18桁ID: YYYYMMDD(8) + 楽天場コード(2) + 000000(6) + RR(2)
    例: 川崎12R 2026-04-07 → 202604072100000012

    日付は実行日（JST）を使用する。キャッシュのrace_dateは古い可能性があるため。
    """
    if not race_id or len(race_id) < 12:
        return ""
    venue_code = race_id[4:6]
    rakuten_venue = RAKUTEN_VENUE_CODE.get(venue_code)
    if not rakuten_venue:
        return ""
    # 実行日（JST）の YYYYMMDD を使用
    from datetime import datetime, timezone, timedelta
    jst = timezone(timedelta(hours=9))
    ymd = datetime.now(jst).strftime("%Y%m%d")
    race_num = race_id[10:12]
    rakuten_id = f"{ymd}{rakuten_venue}000000{race_num}"
    return f"https://bet.keiba.rakuten.co.jp/sp/normal/shikibetu/RACEID/{rakuten_id}"


def _venue_webhook(venue: str, default_url: str) -> str:
    """開催場に対応するWebhook URLを返す。未設定ならdefaultにフォールバック。"""
    env_key = NAR_VENUE_WEBHOOK_MAP.get(venue, "")
    if env_key:
        url = os.environ.get(env_key, "")
        if url:
            logger.info(f"  [webhook] {venue} → {env_key} (設定済み)")
            return url
        else:
            logger.warning(f"  [webhook] {venue} → {env_key} が未設定 → デフォルトにフォールバック")
    else:
        if venue:
            logger.info(f"  [webhook] {venue} → マップに未登録 → デフォルト")
    return default_url

# ── 開催場 → YouTube チャンネル URL ─────────────────────────────
VENUE_YOUTUBE: dict[str, str] = {
    "大井":   "https://www.youtube.com/@tckkeiba",
    "川崎":   "https://www.youtube.com/@kawasakikeiba",
    "船橋":   "https://www.youtube.com/@funabashi-keiba",
    "浦和":   "https://www.youtube.com/@urawa_keiba_official",
    "門別":   "https://www.youtube.com/@hokkaidokeiba",
    "名古屋": "https://www.youtube.com/@nagoyakeiba",
    "園田":   "https://www.youtube.com/@sonodahimejiweb",
    "笠松":   "https://www.youtube.com/@kasamagogo",
    "金沢":   "https://www.youtube.com/@kanazawakeiba_official",
    "高知":   "https://www.youtube.com/@KeibaOrJp",
    "佐賀":   "https://www.youtube.com/@sagakeibaofficial",
}


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# Discord 送信
# ══════════════════════════════════════════════════════════════

def send_discord(webhook_url: str, content: str) -> bool:
    """Discord Webhook にメッセージを送信する。2000 字超は自動分割。"""
    if not webhook_url:
        logger.error("Discord Webhook URL が未設定です")
        return False
    chunks = [content[i : i + 1900] for i in range(0, len(content), 1900)]
    ok = True
    for idx, chunk in enumerate(chunks):
        print(f"[Discord送信] chunk {idx + 1}/{len(chunks)} ({len(chunk)}文字):\n{chunk}", flush=True)
        try:
            # ensure_ascii=False で絵文字(📝等)をUTF-8のまま送信
            payload = json.dumps({"content": chunk}, ensure_ascii=False).encode("utf-8")
            r = requests.post(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=15,
            )
            if r.status_code not in (200, 204):
                logger.error(f"Discord 送信失敗: {r.status_code} {r.text[:200]}")
                ok = False
            else:
                logger.info(f"  Discord 送信OK chunk {idx + 1}/{len(chunks)} ({len(chunk)}文字)")
                time.sleep(1)
        except requests.RequestException as e:
            logger.error(f"Discord 送信エラー: {e}")
            ok = False
    return ok


# ══════════════════════════════════════════════════════════════
# 今週末のレース取得
# ══════════════════════════════════════════════════════════════

def _weekend_dates() -> list[str]:
    """今週末（土・日）の YYYYMMDD リストを返す。月〜土曜実行を想定。"""
    today   = _today_jst()
    wd      = today.weekday()          # 0=月 … 5=土 6=日
    if   wd == 5: d = 0                # 土 → 当日
    elif wd == 6: d = -1               # 日 → 昨日=土
    else:         d = 5 - wd          # 月(4)・火(3)・水(3)・木(2)・金(1) → 今週土
    sat = today + timedelta(days=d)
    sun = sat + timedelta(days=1)
    return [sat.strftime("%Y%m%d"), sun.strftime("%Y%m%d")]


def _detect_grade(el) -> str:
    """BeautifulSoup要素からグレード（"GI"/"GII"/"GIII"）を検出して返す。

    対象クラス（完全一致）:
      icon_gradetype1 → GI
      icon_gradetype2 → GII
      icon_gradetype3 → GIII
    Icon_GradeType16/17/18 などリステッド・オープン・地方重賞はスキップ（""を返す）。
    """
    # クラス名 → グレード文字列のマッピング
    CLASS_GRADE = {"icon_gradetype1": "GI", "icon_gradetype2": "GII", "icon_gradetype3": "GIII"}
    # テキスト/alt → グレード文字列のマッピング（正規表現でマッチ後に判定）
    TEXT_GRADE = {
        re.compile(r"G[Ⅰ1]|GI$"):   "GI",
        re.compile(r"G[Ⅱ2]|GII$"):  "GII",
        re.compile(r"G[Ⅲ3]|GIII$"): "GIII",
    }

    # 1. クラス名で判定（Icon_GradeType1/2/3 のみ。5以上はリステッド等で除外）
    GRADE_TYPE_RE = re.compile(r"^icon_gradetype(\d+)$", re.I)
    for child in el.find_all(True):
        for cls in child.get("class", []):
            m = GRADE_TYPE_RE.match(cls.lower())
            if m:
                num = int(m.group(1))
                if num == 1: return "GI"
                if num == 2: return "GII"
                if num == 3: return "GIII"
                # 4以上（リステッド・オープン等）は重賞ではない
                return ""

    # 2. 旧形式テキストアイコン: gradeicon-g1/g2/g3
    GRADE_CLS_RE = re.compile(r"\bgradeicon-g([123])\b", re.I)
    for child in el.find_all(True):
        cls_str = " ".join(child.get("class", []))
        m = GRADE_CLS_RE.search(cls_str)
        if m:
            return {"1": "GI", "2": "GII", "3": "GIII"}.get(m.group(1), "")

    # 3. 全テキストに括弧付きグレード表記 (G1)/(GⅠ)/(GII)/(GⅡ)/(GIII)/(GⅢ)
    text = el.get_text(" ", strip=True)
    m3 = re.search(r"\(G([Ⅰ1])\)|\(GI\)|\(G([Ⅱ2])\)|\(GII\)|\(G([Ⅲ3])\)|\(GIII\)", text)
    if m3:
        full = m3.group(0)
        if re.search(r"GI{3}|GⅢ|G3", full): return "GIII"
        if re.search(r"GI{2}|GⅡ|G2",  full): return "GII"
        return "GI"

    # 4. 単体テキストが "G1"/"GⅠ" 等の子孫要素
    for child in el.find_all(True):
        stext = child.get_text(strip=True)
        if re.fullmatch(r"G[Ⅲ3]|GIII", stext): return "GIII"
        if re.fullmatch(r"G[Ⅱ2]|GII",  stext): return "GII"
        if re.fullmatch(r"G[Ⅰ1]|GI",   stext): return "GI"

    # 5. 画像 alt 属性
    for img in el.find_all("img", alt=True):
        alt = img["alt"].strip()
        if re.fullmatch(r"G[Ⅲ3]|GIII", alt): return "GIII"
        if re.fullmatch(r"G[Ⅱ2]|GII",  alt): return "GII"
        if re.fullmatch(r"G[Ⅰ1]|GI",   alt): return "GI"

    return ""


def _is_grade_race(el) -> bool:
    """BeautifulSoup要素（<li>など）が GI/GII/GIII かどうかを判定する。"""
    return bool(_detect_grade(el))


def _dump_html_for_debug(soup, kaisai_date: str) -> None:
    """取得した soup の HTML をデバッグ用ファイルに保存する。"""
    try:
        debug_dir = DATA_DIR / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"race_list_{kaisai_date}.html"
        path.write_text(soup.prettify(), encoding="utf-8")
        logger.info(f"  [debug] HTML保存: {path}")
    except Exception as e:
        logger.debug(f"  [debug] HTML保存失敗: {e}")


def scrape_nar_race_ids_for_today(session: requests.Session) -> list[dict]:
    """今日のNARレース一覧を取得する（全会場・全レース）。

    Returns:
        [{"race_id", "race_name", "race_date"}, ...]
    """
    from keiba_predictor.scraper.netkeiba_scraper import (
        scrape_nar_race_ids_for_date, NAR_RACE_LIST_URL, _get,
    )

    today = _today_jst()
    kaisai_date = today.strftime("%Y%m%d")
    race_date_str = today.isoformat()

    logger.info(f"[NAR] 今日のレース取得: {kaisai_date}")
    race_ids = scrape_nar_race_ids_for_date(kaisai_date, session)

    if not race_ids:
        logger.info(f"[NAR] 今日 ({kaisai_date}) の開催なし")
        return []

    # レース名を取得（race_list_sub.html から）
    found: list[dict] = []
    url = f"{NAR_RACE_LIST_URL}?kaisai_date={kaisai_date}"
    soup = _get(url, session, encoding="UTF-8")

    name_map: dict[str, str] = {}
    if soup:
        for a in soup.select("a[href*='race_id=']"):
            m = re.search(r"race_id=(\d{12})", a.get("href", ""))
            if m:
                rid = m.group(1)
                name_el = a.find_parent("li")
                if name_el:
                    n = name_el.select_one(".Race_Name") or name_el.select_one(".RaceName")
                    if n:
                        name_map[rid] = n.get_text(strip=True)

    for rid in race_ids:  # 全レース（会場制限なし）
        found.append({
            "race_id": rid,
            "race_name": name_map.get(rid, rid),
            "race_date": race_date_str,
        })

    logger.info(f"[NAR] 今日のレース: {len(found)} 件")
    return found


def scrape_grade_race_ids(session: requests.Session) -> list[dict]:
    """今週末の重賞レース一覧 [{race_id, race_name, race_date}, ...] を返す。"""
    found: list[dict] = []
    seen:  set[str]   = set()

    dates = _weekend_dates()
    logger.info(f"検索対象日付: {dates[0]} (土) / {dates[1]} (日)")

    # race_list_sub.html（静的フラグメント）と race_list.html の両方を試みる
    LIST_PATHS = ["race_list_sub.html", "race_list.html"]

    for kaisai_date in dates:
        race_date_str = f"{kaisai_date[:4]}-{kaisai_date[4:6]}-{kaisai_date[6:]}"
        found_this_day: list[dict] = []

        for path in LIST_PATHS:
            url = f"https://nar.netkeiba.com/top/{path}?kaisai_date={kaisai_date}"
            logger.info(f"取得中: {url}")
            soup = _get(url, session)
            if soup is None:
                logger.warning(f"  取得失敗: {url}")
                continue

            # ── <li class="RaceList_DataItem"> を起点に取得 ──────────
            # グレードアイコンは <a> タグの外 (<li> 直下) に置かれることが多いため
            # <a> ではなく <li> 全体を検査する
            items = soup.select("li.RaceList_DataItem")
            logger.info(f"  {kaisai_date}: {len(items)} RaceList_DataItem発見 ({path})")

            # 最初の取得時にHTMLをデバッグ保存（クラス構造確認用）
            if items:
                _dump_html_for_debug(soup, kaisai_date)

            for li in items:
                # race_id を li 内の a タグから取得
                a_tag = None
                for a in li.select("a[href]"):
                    if re.search(r"race_id=\d{12}", a.get("href", "")):
                        a_tag = a
                        break
                if a_tag is None:
                    continue
                m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
                if not m:
                    continue
                race_id = m.group(1)
                if race_id in seen:
                    continue

                # JRA競馬場コード（race_id[4:6]）が01〜10のみ対象（NAR は30以上）
                if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                    logger.debug(f"    {race_id} スキップ（NAR venue={race_id[4:6]}）")
                    continue

                # レース名（<li> 全体から複数セレクタで試みる）
                name_el = (
                    li.select_one(".Race_Name")
                    or li.select_one(".RaceName")
                    or li.select_one(".RaceList_ItemTitle")
                    or li.select_one(".ItemTitle")
                )
                race_name = (
                    name_el.get_text(strip=True) if name_el
                    else a_tag.get_text(" ", strip=True)
                )

                # 重賞判定: <li> 全体を渡す（a タグ外のグレードアイコンも検査）
                is_grade = _is_grade_race(li)

                if is_grade:
                    seen.add(race_id)
                    found_this_day.append({
                        "race_id":   race_id,
                        "race_name": race_name,
                        "race_date": race_date_str,
                    })
                    logger.info(f"  ★重賞: {race_name} ({race_id})")

            # フォールバック: <li> がない場合は <a href*='race_id='> から全件取得して
            # <li> と同様に親要素を検査する
            if not items:
                logger.info(f"  RaceList_DataItem なし → href ベースにフォールバック ({path})")
                for a in soup.select("a[href*='race_id=']"):
                    m = re.search(r"race_id=(\d{12})", a.get("href", ""))
                    if not m:
                        continue
                    race_id = m.group(1)
                    if race_id in seen:
                        continue

                    # JRA競馬場コード（race_id[4:6]）が01〜10のみ対象（NAR は30以上）
                    if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                        logger.debug(f"    [fallback] {race_id} スキップ（NAR venue={race_id[4:6]}）")
                        continue

                    # <a> の最も近い block 祖先（<li>/<div>/<tr>）を検査対象にする
                    container = a
                    for anc in a.parents:
                        if anc.name in ("li", "div", "tr", "td"):
                            container = anc
                            break

                    name_el = (
                        container.select_one(".Race_Name")
                        or container.select_one(".RaceName")
                        or container.select_one(".RaceList_ItemTitle")
                    )
                    race_name = (
                        name_el.get_text(strip=True) if name_el
                        else a.get_text(" ", strip=True)
                    )

                    is_grade = _is_grade_race(container)
                    if is_grade:
                        seen.add(race_id)
                        found_this_day.append({
                            "race_id":   race_id,
                            "race_name": race_name,
                            "race_date": race_date_str,
                        })
                        logger.info(f"  ★重賞(fallback): {race_name} ({race_id})")

            # 重賞が見つかれば次のURLは試さない
            if found_this_day:
                break
            if items:
                # アイテムはあったが重賞なし → もう一方のURLも試す
                logger.info(f"  {path}: {len(items)}件あるが重賞0件 → 次URLを試みる")

        found.extend(found_this_day)
        _sleep()

    logger.info(f"重賞合計: {len(found)} レース")
    return found


def update_featured_races_csv(
    path: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> int:
    """翌週末（土日）の重賞レースを netkeiba からスクレイピングし、
    featured_races.csv（形式: race_id,race_name,grade）に上書き保存する。

    Returns:
        保存したレース数（0 の場合はスクレイピング失敗 or 重賞なし）
    """
    if path is None:
        path = DATA_DIR / "featured_races.csv"
    if session is None:
        session = requests.Session()

    dates = _weekend_dates()
    logger.info(f"[update_featured] 対象日付: {dates[0]} (土) / {dates[1]} (日)")

    LIST_PATHS = ["race_list_sub.html", "race_list.html"]
    found: list[dict] = []
    seen: set[str] = set()

    for kaisai_date in dates:
        found_this_day: list[dict] = []

        for list_path in LIST_PATHS:
            url = f"https://nar.netkeiba.com/top/{list_path}?kaisai_date={kaisai_date}"
            logger.info(f"[update_featured] 取得中: {url}")
            soup = _get(url, session)
            if soup is None:
                logger.warning(f"[update_featured] 取得失敗: {url}")
                continue

            items = soup.select("li.RaceList_DataItem")
            logger.info(f"[update_featured] {kaisai_date}: {len(items)} アイテム ({list_path})")

            for li in items:
                a_tag = None
                for a in li.select("a[href]"):
                    if re.search(r"race_id=\d{12}", a.get("href", "")):
                        a_tag = a
                        break
                if a_tag is None:
                    continue
                m = re.search(r"race_id=(\d{12})", a_tag.get("href", ""))
                if not m:
                    continue
                race_id = m.group(1)
                if race_id in seen:
                    continue
                # JRA 競馬場コードのみ（NAR はスキップ）
                if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                    continue

                name_el = (
                    li.select_one(".Race_Name")
                    or li.select_one(".RaceName")
                    or li.select_one(".RaceList_ItemTitle")
                    or li.select_one(".ItemTitle")
                )
                race_name = (
                    name_el.get_text(strip=True) if name_el
                    else a_tag.get_text(" ", strip=True)
                )

                grade = _detect_grade(li)
                logger.debug(f"[update_featured]   {race_id} [{race_name!r}] grade={grade!r}")

                if grade:
                    seen.add(race_id)
                    found_this_day.append({
                        "race_id":   race_id,
                        "race_name": race_name,
                        "grade":     grade,
                    })
                    logger.info(f"[update_featured] ★ {grade} {race_name} ({race_id})")

            # フォールバック: RaceList_DataItem がない場合
            if not items:
                for a in soup.select("a[href*='race_id=']"):
                    m = re.search(r"race_id=(\d{12})", a.get("href", ""))
                    if not m:
                        continue
                    race_id = m.group(1)
                    if race_id in seen:
                        continue
                    if race_id[4:6] not in {"01","02","03","04","05","06","07","08","09","10"}:
                        continue
                    container = a
                    for anc in a.parents:
                        if anc.name in ("li", "div", "tr", "td"):
                            container = anc
                            break
                    name_el = (
                        container.select_one(".Race_Name")
                        or container.select_one(".RaceName")
                        or container.select_one(".RaceList_ItemTitle")
                    )
                    race_name = (
                        name_el.get_text(strip=True) if name_el
                        else a.get_text(" ", strip=True)
                    )
                    grade = _detect_grade(container)
                    if grade:
                        seen.add(race_id)
                        found_this_day.append({
                            "race_id":   race_id,
                            "race_name": race_name,
                            "grade":     grade,
                        })
                        logger.info(f"[update_featured] ★(fallback) {grade} {race_name} ({race_id})")

            if found_this_day:
                break

        found.extend(found_this_day)
        _sleep()

    if not found:
        logger.warning("[update_featured] 重賞レースが見つかりませんでした。featured_races.csv は更新しません。")
        return 0

    # CSV 保存
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{r['race_id']},{r['race_name']},{r['grade']}" for r in found]
    path.write_text("race_id,race_name,grade\n" + "\n".join(rows) + "\n", encoding="utf-8-sig")
    logger.info(f"[update_featured] featured_races.csv 保存完了: {len(found)} レース → {path}")
    return len(found)


def _save_upcoming_to_cache() -> None:
    """featured_races.csv のレース情報を predictions_cache.json に upcoming として保存する。

    既存の予想データがあるレースは上書きしない。
    """
    featured_path = DATA_DIR / "featured_races.csv"
    if not featured_path.exists():
        logger.warning("featured_races.csv が見つかりません")
        return

    try:
        df = pd.read_csv(featured_path, encoding="utf-8-sig", dtype={"race_id": str})
    except Exception as e:
        logger.warning(f"featured_races.csv 読み込み失敗: {e}")
        return

    cache = _load_cache()
    dates = _weekend_dates()

    VENUE_MAP = {
        # JRA
        "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
        "05": "東京", "06": "中山", "07": "中京", "08": "京都",
        "09": "阪神", "10": "小倉",
        # NAR
        "30": "門別", "31": "帯広",
        "35": "盛岡", "36": "水沢",
        "42": "浦和", "43": "船橋", "44": "大井", "45": "川崎",
        "46": "金沢", "47": "笠松", "48": "名古屋",
        "50": "園田", "51": "姫路",
        "54": "高知", "55": "佐賀",
    }

    added = 0
    for _, row in df.iterrows():
        race_id = str(row["race_id"])
        # 既に予想データがあるレースはスキップ
        if race_id in cache and cache[race_id].get("predicted_top3_nums"):
            continue

        race_date_str = ""
        if len(dates) >= 2:
            # race_id から土日を判定（末尾2桁がレース番号、その前が日次）
            race_date_str = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"

        venue_code = race_id[4:6] if len(race_id) >= 6 else ""
        venue = VENUE_MAP.get(venue_code, "")

        cache[race_id] = {
            "race_name":           str(row.get("race_name", race_id)),
            "race_date":           race_date_str,
            "start_time":          "",
            "venue":               venue,
            "course_info":         "",
            "honmei":              None,
            "taikou":              None,
            "ana":                 None,
            "predicted_top3_nums": [],
            "predicted_top5_nums": [],
            "predicted_top5":      [],
            "ev_top3":             [],
            "dangerous_horses":    [],
            "ai_comments":         {},
            "status":              "upcoming",
        }
        added += 1

    _save_cache(cache)
    logger.info(f"upcoming レース {added} 件をキャッシュに保存（既存 {len(cache) - added} 件は維持）")


def _load_featured_race_ids_for_weekend(
    featured_path: Optional[Path] = None,
) -> list[dict]:
    """
    featured_races.csv から今週末（土日）の日付に一致するレースIDを返す。

    scrape_grade_race_ids() のフォールバック用。
    今週末のレースIDを手動で featured_races.csv に登録しておくことで
    スクレイピング失敗時でも予想が動くようになる。

    Returns:
        [{"race_id": str, "race_name": str, "race_date": str}, ...]
    """
    if featured_path is None:
        featured_path = DATA_DIR / "featured_races.csv"
    if not featured_path.exists():
        logger.warning(f"featured_races.csv が見つかりません: {featured_path}")
        return []

    try:
        df = pd.read_csv(featured_path, encoding="utf-8-sig", dtype={"race_id": str})
    except Exception as e:
        logger.warning(f"featured_races.csv 読み込み失敗: {e}")
        return []

    if "race_id" not in df.columns:
        return []

    # race_date 列がない新フォーマット（race_id, race_name, grade）の場合は全件返す
    if "race_date" not in df.columns:
        dates = _weekend_dates()
        sat = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}"
        result = []
        for _, row in df.drop_duplicates(subset=["race_id"]).iterrows():
            result.append({
                "race_id":   str(row["race_id"]),
                "race_name": str(row.get("race_name", row["race_id"])),
                "race_date": sat,
            })
        if result:
            logger.info(
                f"[featured fallback] {len(result)} レース "
                f"({', '.join(r['race_name'] for r in result)})"
            )
        return result

    dates = _weekend_dates()  # ["YYYYMMDD", "YYYYMMDD"]
    weekend_dates = {
        f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in dates
    }

    mask = df["race_date"].astype(str).str[:10].isin(weekend_dates)
    weekend_df = df[mask].drop_duplicates(subset=["race_id"])

    result = []
    for _, row in weekend_df.iterrows():
        result.append({
            "race_id":   str(row["race_id"]),
            "race_name": str(row.get("race_name", row["race_id"])),
            "race_date": str(row["race_date"])[:10],
        })

    if result:
        logger.info(
            f"[featured fallback] {len(result)} レース "
            f"({', '.join(r['race_name'] for r in result)})"
        )
    return result


# ══════════════════════════════════════════════════════════════
# 予想キャッシュ
# ══════════════════════════════════════════════════════════════

def _load_cache() -> dict:
    """予想キャッシュを読み込む。当日のスナップショットがあればそちらを優先する。"""
    snapshot_path = DATA_DIR / f"predictions_snapshot_{_today_jst().strftime('%Y%m%d')}.json"
    if snapshot_path.exists():
        logger.info(f"スナップショットを使用: {snapshot_path.name}")
        try:
            with open(snapshot_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"スナップショットの読み込みに失敗: {e}")
    if PRED_CACHE.exists():
        try:
            with open(PRED_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"予想キャッシュの読み込みに失敗: {e}")
    return {}


def _save_cache(cache: dict) -> None:
    PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRED_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    # ── 書き込み確認ログ ──────────────────────────────────────
    size = PRED_CACHE.stat().st_size
    keys = list(cache.keys())
    print(f"[_save_cache] 書き込み完了: {PRED_CACHE.resolve()} ({size}bytes, {len(keys)}件: {keys})", flush=True)


def _ana_horse_info(result_df: pd.DataFrame, ana_horse_num: "Optional[int]") -> dict:
    """穴馬の詳細情報を返す（キャッシュ保存用）。"""
    if ana_horse_num is None or result_df.empty:
        return {}
    match = result_df[pd.to_numeric(result_df["horse_number"], errors="coerce") == ana_horse_num]
    if match.empty:
        return {}
    r = match.iloc[0]
    return {
        "horse_number": ana_horse_num,
        "horse_name": str(r.get("horse_name", "")),
        "prob": round(float(r.get("prob_top3", 0)), 4),
        "popularity": int(pd.to_numeric(r.get("popularity"), errors="coerce") or 0),
    }


def _store_prediction(race_id: str, race_name: str, race_date: str,
                      result_df: pd.DataFrame,
                      ai_comments: Optional[dict] = None,
                      course_info: str = "",
                      start_time: str = "",
                      venue: str = "") -> None:
    """予想結果をキャッシュに保存する（日曜結果比較・note レポート生成に使用）。"""
    cache = _load_cache()

    # オッズが全馬同一（仮オッズ）かチェック
    _odds_ser = pd.to_numeric(result_df["odds"], errors="coerce").dropna()
    _all_same = len(_odds_ser) > 1 and _odds_ser.nunique() == 1

    def _row(df: pd.DataFrame, idx: int) -> dict:
        if len(df) <= idx:
            return {}
        r = df.iloc[idx]
        raw_o = pd.to_numeric(r.get("odds"), errors="coerce")
        o_val = None if (not pd.notna(raw_o) or _all_same) else round(float(raw_o), 1)
        entry = {
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "prob":         float(r["prob_top3"]),
            "odds":         o_val,
        }
        # SHAP値の上位要因を追加
        shap_top = r.get("shap_top")
        if isinstance(shap_top, list) and shap_top:
            entry["shap_top"] = shap_top
        return entry

    # 穴馬: AI確率35%以上 & 6番人気以下 & TOP3外 → AI確率最高の1頭
    ana: dict = {}
    try:
        top3_set = set()
        for _, r in result_df.head(3).iterrows():
            v = r.get("horse_number")
            if pd.notna(v):
                top3_set.add(int(v))
        rest = result_df.iloc[3:] if len(result_df) > 3 else pd.DataFrame()
        if not rest.empty:
            rest_prob = pd.to_numeric(rest["prob_top3"], errors="coerce")
            rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
            cands = rest[(rest_prob >= 0.35) & (rest_pop >= 6)]
            if not cands.empty:
                best_idx = cands["prob_top3"].idxmax()
                ana = _row(result_df, result_df.index.get_loc(best_idx))
    except Exception as e:
        logger.warning(f"穴馬検出失敗: {e}")
    if not ana:
        ana = _row(result_df, 2)

    top3_nums = []
    for _, row in result_df.head(3).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top3_nums.append(int(v))

    top5_nums = []
    for _, row in result_df.head(5).iterrows():
        v = row.get("horse_number")
        if pd.notna(v):
            top5_nums.append(int(v))

    # 穴馬（3連複用）: AI確率35%以上 & 6番人気以下 & TOP5外 → AI確率最高
    ana_horse_num: Optional[int] = None
    if len(result_df) > 5:
        rest = result_df.iloc[5:]
        rest_prob = pd.to_numeric(rest["prob_top3"], errors="coerce")
        rest_pop = pd.to_numeric(rest.get("popularity", pd.Series(dtype=float)), errors="coerce")
        cands = rest[(rest_prob >= 0.35) & (rest_pop >= 6)]
        if not cands.empty:
            best = cands.nlargest(1, "prob_top3").iloc[0]
            v = best.get("horse_number")
            if pd.notna(v):
                ana_horse_num = int(v)

    # EV・危険馬データ（_calc_ev_and_flags 済みならそのまま使う）
    if "ev_score" not in result_df.columns:
        result_df = calc_ev_and_flags(result_df)

    # オッズが全馬同一（仮オッズ）かチェック
    odds_vals = pd.to_numeric(result_df["odds"], errors="coerce").dropna()
    all_same_odds = len(odds_vals) > 1 and odds_vals.nunique() == 1
    if all_same_odds:
        logger.warning(f"全馬同一オッズ({odds_vals.iloc[0]}) → 仮オッズのため odds=None として保存")

    ev_top3: list[dict] = []
    for _, r in result_df[result_df["ev_score"].notna()].nlargest(3, "ev_score").iterrows():
        raw_odds = pd.to_numeric(r.get("odds"), errors="coerce")
        odds_val = None if (not pd.notna(raw_odds) or all_same_odds) else round(float(raw_odds), 1)
        ev_top3.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "ev_score":     round(float(r["ev_score"]), 3),
            "prob":         round(float(r["prob_top3"]), 4),
            "odds":         odds_val,
        })

    dangerous: list[dict] = []
    for _, r in result_df[result_df["is_dangerous"]].iterrows():
        dangerous.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "popularity":   int(pd.to_numeric(r.get("popularity"), errors="coerce") or 0),
            "prob":         round(float(r["prob_top3"]), 4),
            "reasons":      list(r.get("danger_reasons", [])),
        })

    # 上位5頭の詳細情報（Discord通知用）
    predicted_top5: list[dict] = []
    for i in range(min(5, len(result_df))):
        r = result_df.iloc[i]
        raw_o = pd.to_numeric(r.get("odds"), errors="coerce")
        o_val = None if (not pd.notna(raw_o) or _all_same) else round(float(raw_o), 1)
        predicted_top5.append({
            "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else None,
            "horse_name":   str(r.get("horse_name", "")),
            "prob":         round(float(r["prob_top3"]), 4),
            "odds":         o_val,
        })

    cache[race_id] = {
        "race_name":           race_name,
        "race_date":           race_date,
        "start_time":          start_time,
        "venue":               venue,
        "course_info":         course_info,
        "honmei":              _row(result_df, 0),
        "taikou":              _row(result_df, 1),
        "ana":                 ana,
        "predicted_top3_nums": top3_nums,
        "predicted_top5_nums": top5_nums,
        "ana_horse_num":       ana_horse_num,
        "ana_horse_info":      _ana_horse_info(result_df, ana_horse_num),
        "predicted_top5":      predicted_top5,
        "ev_top3":             ev_top3,
        "dangerous_horses":    dangerous,
        "ai_comments":         ai_comments or {},
    }

    # 買い目自動決定
    try:
        from keiba_predictor.model.predict import _decide_bet_strategy
        cache[race_id]["bet_strategy"] = _decide_bet_strategy(result_df)
    except Exception as e:
        logger.warning(f"買い目自動決定失敗: {e}")

    # モンテカルロシミュレーション
    try:
        from keiba_predictor.simulation import run_monte_carlo
        mc_horses = []
        for i in range(min(len(result_df), 18)):
            r = result_df.iloc[i]
            mc_horses.append({
                "horse_number": int(r["horse_number"]) if pd.notna(r.get("horse_number")) else i + 1,
                "horse_name": str(r.get("horse_name", "")),
                "prob": float(r["prob_top3"]),
                "running_style_enc": int(r.get("running_style_enc", 2)) if pd.notna(r.get("running_style_enc")) else 2,
            })
        mc_result = run_monte_carlo(mc_horses)
        # 上位5頭のみ保存
        top5_mc = {}
        for num in top5_nums[:5]:
            k = str(num)
            if k in mc_result:
                top5_mc[k] = mc_result[k]
        cache[race_id]["simulation"] = top5_mc
        # MC確率でキャッシュ内のprobを上書き（Discord・ダッシュボード表示用）
        for role in ("honmei", "taikou", "ana"):
            p = cache[race_id].get(role, {})
            if p and p.get("horse_number") is not None:
                mc = mc_result.get(str(p["horse_number"]), {})
                if "top3_rate" in mc:
                    p["prob"] = mc["top3_rate"]
        for h in cache[race_id].get("predicted_top5", []):
            if h.get("horse_number") is not None:
                mc = mc_result.get(str(h["horse_number"]), {})
                if "top3_rate" in mc:
                    h["prob"] = mc["top3_rate"]
        # MC確率ベースで危険馬を再判定
        new_danger = []
        for d in cache[race_id].get("dangerous_horses", []):
            num = d.get("horse_number")
            mc = mc_result.get(str(num), {})
            mc_prob = mc.get("top3_rate")
            pop = d.get("popularity")
            if mc_prob is not None and pop is not None and pop <= 3 and mc_prob >= 0.40:
                # MC確率40%以上なら危険解除
                logger.info(f"  危険解除（MC {mc_prob*100:.0f}%）: {d.get('horse_name')} {pop}人気")
                continue
            new_danger.append(d)
        cache[race_id]["dangerous_horses"] = new_danger
    except Exception as e:
        logger.warning(f"モンテカルロシミュレーション失敗: {e}")

    # モデル情報をキャッシュに保持（ダッシュボード表示用）
    cache["_model_metrics"] = {
        "auc": 0.8162,
        "fukusho_rate": 60.5,
        "n_features": 44,
    }

    _save_cache(cache)


# ══════════════════════════════════════════════════════════════
# 払戻金取得
# ══════════════════════════════════════════════════════════════

def scrape_payouts(race_id: str, session: requests.Session) -> dict:
    """レース払戻金を取得する。

    Returns:
        {"馬連": [{"combo": "3-5", "amount": 1450}], "ワイド": [...], ...}
    """
    # NAR: result.html?race_id=...  (EUC-JP)
    url = f"{NAR_RESULT_URL}?race_id={race_id}"
    soup = _get(url, session, encoding="euc-jp")
    if soup is None:
        return {}

    payouts: dict[str, list] = {}

    def _parse_yen(s: str) -> Optional[int]:
        s = re.sub(r"[¥￥,円\s]", "", s)
        try:
            return int(s)
        except ValueError:
            return None

    # NAR: Payout_Detail_Table / JRA: pay_table_01
    # NAR形式: 1行に複数エントリが連結される（例: combo="46", amount="130円190円"）
    # td内のbr/spanで分割されている場合もある
    pay_tables = soup.select("table.Payout_Detail_Table, table.pay_table_01")
    for table in pay_tables:
        current_type = None
        for tr in table.select("tr"):
            th = tr.select_one("th")
            tds = tr.select("td")
            if th:
                current_type = th.get_text(strip=True)
            if not current_type or len(tds) < 2:
                continue

            # brタグを改行に変換して分割（JRA: <br>区切り / NAR: span/div区切り）
            combo_parts = [p.strip() for p in tds[0].get_text("\n").split("\n") if p.strip()]
            amt_parts   = [p.strip() for p in tds[1].get_text("\n").split("\n") if p.strip()]
            amt_list = [_parse_yen(a) for a in amt_parts]

            n_amt = len(amt_list)
            n_combo = len(combo_parts)

            if n_combo == n_amt and n_amt > 0:
                # 1対1マッチ（複勝: 3馬番 vs 3金額）
                for combo, amt in zip(combo_parts, amt_list):
                    payouts.setdefault(current_type, []).append({
                        "combo": combo, "amount": amt,
                    })
            elif n_amt > 0 and n_combo > n_amt and n_combo % n_amt == 0:
                # combo を n_amt 個のグループに等分割（ワイド: 6馬番 → 3組×2馬番）
                group_size = n_combo // n_amt
                for i, amt in enumerate(amt_list):
                    group = combo_parts[i * group_size:(i + 1) * group_size]
                    combo_str = "-".join(group)
                    payouts.setdefault(current_type, []).append({
                        "combo": combo_str, "amount": amt,
                    })
            elif n_amt > 0:
                # それ以外: 全体を連結して各金額に対応
                combo_all = "-".join(combo_parts)
                for amt in amt_list:
                    payouts.setdefault(current_type, []).append({
                        "combo": combo_all, "amount": amt,
                    })

    if payouts:
        logger.info(f"  払戻金取得: {race_id} → {list(payouts.keys())}")
    else:
        logger.warning(f"  払戻金テーブルなし: {race_id}")

    return payouts




def _record_manual_result(race_id: str, race_name: str, race_date: str,
                          pred: dict, manual: dict) -> None:
    """manual_results.json の的中フラグで results_history.csv に記録する。"""
    from keiba_predictor.history import HISTORY_PATH, DATA_DIR, _grade_label, _pred_row

    # レース名: manual 優先 → 引数 → race_id
    name = manual.get("race_name") or race_name or race_id
    grade = _grade_label(name)
    p1 = _pred_row(pred, "honmei")
    p2 = _pred_row(pred, "taikou")
    p3 = _pred_row(pred, "ana")

    result_nums = manual.get("result", [])
    manual_pay = manual.get("payouts", {})

    fukusho_hit = manual.get("fukusho_hit", False)
    umaren_hit  = manual.get("umaren_hit", False)
    sanren_hit  = manual.get("sanrenpuku_hit", False)

    fukusho_payout  = manual_pay.get("fukusho", 0)
    umaren_payout   = manual_pay.get("umaren", 0)
    sanren_payout   = manual_pay.get("sanrenpuku", 0)
    # 投資: 複勝1点 + 馬連3点 + 3連複10点 = 14点 × 100円
    bet_total       = 1400
    return_total    = fukusho_payout + umaren_payout + sanren_payout

    def _a(i):
        return {"name": "", "num": result_nums[i] if i < len(result_nums) else 0}

    row = {
        "date":       race_date,
        "race_id":    race_id,
        "race_name":  name,
        "race_grade": grade,
        "pred1_name": p1["name"], "pred1_num": p1["num"], "pred1_prob": p1["prob"],
        "pred2_name": p2["name"], "pred2_num": p2["num"], "pred2_prob": p2["prob"],
        "pred3_name": p3["name"], "pred3_num": p3["num"], "pred3_prob": p3["prob"],
        "actual1_name": _a(0)["name"], "actual1_num": _a(0)["num"],
        "actual2_name": _a(1)["name"], "actual2_num": _a(1)["num"],
        "actual3_name": _a(2)["name"], "actual3_num": _a(2)["num"],
        "fukusho_hit":     fukusho_hit,
        "umaren_hit":      umaren_hit,     "umaren_payout":   umaren_payout,
        "wide_hit":        False,          "wide_payout":     0,
        "sanrenpuku_hit":  sanren_hit,     "sanrenpuku_payout": sanren_payout,
        "bet_total":       bet_total,
        "return_total":    return_total,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_row_df = pd.DataFrame([row])
    if HISTORY_PATH.exists():
        new_row_df.to_csv(HISTORY_PATH, mode="a", header=False,
                          index=False, encoding="utf-8-sig")
    else:
        new_row_df.to_csv(HISTORY_PATH, mode="w", header=True,
                          index=False, encoding="utf-8-sig")
    logger.info(f"  [history] 手動記録: {name} fukusho={fukusho_hit} umaren={umaren_hit} sanren={sanren_hit} return=¥{return_total:,}")


CELEBRATION_GIFS = [
    "https://media.tenor.com/x8v1oNUOmg4AAAAC/celebrate.gif",
    "https://media.tenor.com/ZEcGIEKMiZkAAAAC/horse-racing-win.gif",
    "https://media.tenor.com/LkHCEimYMmEAAAAC/confetti.gif",
    "https://media.tenor.com/vGMbFl7vdhAAAAAC/winning.gif",
]


def _send_hit_embed(webhook_url: str, embed: dict) -> bool:
    """Discord Webhook に embed + GIF を送信する。"""
    if not webhook_url:
        return False
    payload = json.dumps({"embeds": [embed]}, ensure_ascii=False).encode("utf-8")
    try:
        r = requests.post(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=15,
        )
        if r.status_code in (200, 204):
            logger.info(f"  的中embed送信OK")
            return True
        else:
            logger.error(f"  的中embed送信失敗: {r.status_code} {r.text[:200]}")
            return False
    except requests.RequestException as e:
        logger.error(f"  的中embed送信エラー: {e}")
        return False


def _build_hit_embed(
    venue: str,
    race_name: str,
    honmei_num: Optional[int],
    honmei_name: str,
    wide_hit: bool,
    wide_pay: str,
    sanren_hit: bool = False,
    sanren_pay: str = "",
    race_id: str = "",
) -> Optional[dict]:
    """的中時のembed辞書を生成する。何も的中していなければ None。"""
    if not wide_hit:
        return None

    # レース番号をrace_idから取得（末尾2桁）
    race_num = ""
    if race_id and len(race_id) >= 12:
        try:
            race_num = f"{int(race_id[10:12])}R "
        except ValueError:
            pass

    # 説明テキスト
    lines = [f"🏇 **{venue} {race_num}{race_name}**"]
    if honmei_num is not None:
        lines.append(f"◎ {honmei_num}番 {honmei_name}")
    lines.append("")

    if wide_hit:
        detail = "ワイド ✅ 的中"
        if wide_pay:
            detail += f"（配当{re.sub(r'[¥,]', '', str(wide_pay))}円）"
        lines.append(detail)

    color = 0x2ECC71

    embed = {
        "title": "🎯 NAR的中！",
        "description": "\n".join(lines),
        "color": color,
        "image": {"url": random.choice(CELEBRATION_GIFS)},
        "footer": {"text": "KEIBA EDGE — AI地方競馬予想"},
    }

    # YouTube動画リンクをfieldsに追加
    yt_url = VENUE_YOUTUBE.get(venue, "")
    if yt_url:
        embed["fields"] = [
            {"name": "📹 レース動画", "value": yt_url, "inline": False}
        ]

    return embed


def _fmt_result(race_name: str, race_date: str,
                actual_df: pd.DataFrame,
                pred: dict,
                payouts: dict,
                manual: Optional[dict] = None,
                race_id: str = "") -> str:
    """日曜結果メッセージを生成する。"""
    RULE = "━" * 24
    venue = pred.get("venue", "")
    race_num = ""
    if race_id and len(race_id) >= 12:
        try:
            race_num = f"{int(race_id[10:12])}R "
        except ValueError:
            pass
    header = f"{venue} {race_num}{race_name}".strip()
    lines = [f"🏆 【KEIBA EDGE】{header} 結果  {race_date}", RULE]

    # 予想馬番→印 のマッピング
    pred_num_to_mark: dict[int, str] = {}
    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "△")]:
        p = pred.get(role, {})
        num = p.get("horse_number")
        if num is not None:
            pred_num_to_mark[int(num)] = mark

    # manual_results.json の predicted_top3_nums があれば優先
    predicted_nums = pred.get("predicted_top3_nums", [])
    if manual and manual.get("predicted_top3_nums"):
        predicted_nums = manual["predicted_top3_nums"]

    # 馬番→馬名マップを予想キャッシュから構築（結果に馬名がない場合の補完用）
    num_to_name: dict[int, str] = {}
    for role in ("honmei", "taikou", "ana"):
        p = pred.get(role, {})
        pnum = p.get("horse_number")
        if pnum is not None:
            num_to_name[int(pnum)] = p.get("horse_name", "")
    for h in (pred.get("predicted_top5") or []):
        hnum = h.get("horse_number")
        if hnum is not None and int(hnum) not in num_to_name:
            num_to_name[int(hnum)] = h.get("horse_name", "")
    for e in (pred.get("ev_top3") or []):
        enum = e.get("horse_number")
        if enum is not None and int(enum) not in num_to_name:
            num_to_name[int(enum)] = e.get("horse_name", "")
    # actual_df からも馬名を補完
    for _, r in actual_df.iterrows():
        anum = r.get("horse_number")
        aname = str(r.get("horse_name", ""))
        if pd.notna(anum) and aname and int(anum) not in num_to_name:
            num_to_name[int(anum)] = aname

    # 確定 1〜3 着
    df_copy = actual_df.copy()
    df_copy["_fp"] = pd.to_numeric(df_copy["finish_position"], errors="coerce")
    top3 = df_copy[df_copy["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)

    actual_top3_nums: list[int] = []
    for _, r in top3.iterrows():
        fp   = int(r["_fp"])
        num  = int(r["horse_number"]) if pd.notna(r.get("horse_number")) else 0
        name = str(r.get("horse_name", ""))
        if not name:
            name = num_to_name.get(num, "")
        actual_top3_nums.append(num)
        mark = pred_num_to_mark.get(num, "　")
        icon = " ✅" if num in predicted_nums else ""
        lines.append(f"{fp}着 {mark} {num}番 {name}{icon}")

    lines.append(RULE)

    # honmei: manual > predicted_nums[0] > pred["honmei"]
    if manual and manual.get("honmei") is not None:
        honmei_num = int(manual["honmei"])
    elif predicted_nums:
        honmei_num = predicted_nums[0]
    else:
        honmei_num = pred.get("honmei", {}).get("horse_number")
    honmei_name = num_to_name.get(honmei_num, "") if honmei_num else ""

    # manual_results.json のフラグがあればそちらを優先
    fukusho_pay = ""
    wide_hit = False
    wide_pay = ""
    if manual and "fukusho_hit" in manual:
        fukusho_hit = manual["fukusho_hit"]
        umaren_hit  = manual.get("umaren_hit", False)
        sanren_hit  = manual.get("sanrenpuku_hit", False)
        wide_hit    = manual.get("wide_hit", False)
        manual_pay  = manual.get("payouts", {})
        if manual_pay.get("fukusho"):
            fukusho_pay = f"{manual_pay['fukusho']:,}"
        umaren_pay  = f"¥{manual_pay['umaren']:,}" if manual_pay.get("umaren") else ""
        sanren_pay  = f"¥{manual_pay['sanrenpuku']:,}" if manual_pay.get("sanrenpuku") else ""
        wide_pay    = f"¥{manual_pay['wide']:,}" if manual_pay.get("wide") else ""
    else:
        # bet_strategy があればそれに基づいて判定
        bs = pred.get("bet_strategy", {})
        actual_set = set(actual_top3_nums[:3]) if len(actual_top3_nums) >= 3 else set()

        # 複勝
        fukusho_hit = (honmei_num is not None) and (int(honmei_num) in actual_top3_nums)
        if fukusho_hit and honmei_num:
            for entry in payouts.get("複勝", []):
                combo_nums = set(re.findall(r"\d+", str(entry.get("combo", ""))))
                if str(honmei_num) in combo_nums:
                    amt = entry.get("amount")
                    if amt:
                        fukusho_pay = f"{amt:,}" if isinstance(amt, int) else re.sub(r"[¥￥,円\s]", "", str(amt))
                    break

        # 馬連（bet_strategy の umaren を使用）
        umaren_hit = False
        umaren_pay = ""
        if bs.get("umaren"):
            actual_12 = set(actual_top3_nums[:2]) if len(actual_top3_nums) >= 2 else set()
            for u in bs["umaren"]:
                if set(u["nums"]) == actual_12:
                    combo = f"{u['nums'][0]}-{u['nums'][1]}"
                    umaren_pay = _get_payout(payouts, "馬連", combo)
                    umaren_hit = True
                    break
        if not umaren_hit:
            umaren_hit, umaren_pay = _check_umaren_raw(predicted_nums, actual_top3_nums, payouts)

        # ワイド（bet_strategy の wide を使用）
        wide_hit = False
        wide_pay = ""
        if bs.get("wide"):
            for w in bs["wide"]:
                a, b = w["nums"]
                if a in actual_set and b in actual_set:
                    combo = f"{a}-{b}"
                    wide_pay = _get_payout(payouts, "ワイド", combo)
                    wide_hit = True
                    break

        # 3連複（bet_strategy の sanrenpuku を使用）
        sanren_hit = False
        sanren_pay = ""
        sr = bs.get("sanrenpuku", {})
        if sr and sr.get("jiku") and sr.get("aite") and len(actual_set) >= 3:
            jiku = sr["jiku"]
            aite = sr["aite"]
            if len(jiku) == 1 and jiku[0] in actual_set:
                for pair in combinations(aite, 2):
                    if {jiku[0], pair[0], pair[1]} == actual_set:
                        combo = "-".join(str(n) for n in sorted([jiku[0], pair[0], pair[1]]))
                        sanren_pay = _get_payout(payouts, "三連複", combo)
                        sanren_hit = True
                        break
            elif len(jiku) == 2 and jiku[0] in actual_set and jiku[1] in actual_set:
                for a in aite:
                    if {jiku[0], jiku[1], a} == actual_set:
                        combo = "-".join(str(n) for n in sorted([jiku[0], jiku[1], a]))
                        sanren_pay = _get_payout(payouts, "三連複", combo)
                        sanren_hit = True
                        break
        if not sanren_hit:
            # フォールバック
            ana_horse_num = pred.get("ana_horse_num")
            sanren_hit, sanren_pay = _check_sanrenpuku_raw(predicted_nums, actual_top3_nums, payouts, ana_horse_num, pred=pred)

    wide_line = f"ワイド {'✅ 的中' if wide_hit else '❌ ハズレ'}"
    if wide_hit and wide_pay:
        wide_line += f"（配当{re.sub(r'[¥,]', '', str(wide_pay))}円）"
    lines.append(wide_line)

    return "\n".join(lines)


def _get_payout(payouts: dict, bet_type: str, combo: str) -> str:
    """払戻金辞書から指定の組み合わせ・金額を文字列で返す。"""
    # netkeibaのHTMLでは "3連複"(半角) と "三連複"(漢数字) が混在するため両方試す
    _ALIASES = {
        "三連複": ["三連複", "3連複"],
        "三連単": ["三連単", "3連単"],
        "馬連": ["馬連"],
        "馬単": ["馬単"],
        "ワイド": ["ワイド"],
        "単勝": ["単勝"],
        "複勝": ["複勝"],
    }
    keys = _ALIASES.get(bet_type, [bet_type])
    for key in keys:
        for entry in payouts.get(key, []):
            e_nums = set(re.findall(r"\d+", entry["combo"]))
            c_nums = set(re.findall(r"\d+", combo))
            if e_nums == c_nums:
                amt = entry["amount"]
                return f"¥{amt:,}" if amt else ""
    return ""


# ── 買い目判定（module-level: history.py からも呼び出せる） ────────────

def _check_umaren_raw(
    predicted_nums: list[int],
    actual_top3_nums: list[int],
    payouts: dict,
) -> tuple[bool, str]:
    """馬連的中判定。買い目はtop3の全組み合わせ(3点)。(hit, pay_str) を返す。"""
    if len(predicted_nums) < 2 or len(actual_top3_nums) < 2:
        return False, ""
    a1, a2 = actual_top3_nums[0], actual_top3_nums[1]
    actual_set = {a1, a2}
    # 買い目: predicted_top3_nums[:3] の全組み合わせ
    for pair in combinations(predicted_nums[:3], 2):
        if set(pair) == actual_set:
            combo = f"{pair[0]}-{pair[1]}"
            pay = _get_payout(payouts, "馬連", combo)
            return True, pay
    return False, ""


def _check_wide_pairs_raw(
    predicted_nums: list[int],
    actual_top3_nums: list[int],
    payouts: dict,
) -> list[tuple[str, bool, str]]:
    """ワイド全組み合わせ判定。[(combo, hit, pay_str), ...] を返す。"""
    results = []
    if len(predicted_nums) < 2 or len(actual_top3_nums) < 3:
        return results
    for a, b in combinations(predicted_nums[:3], 2):
        hit   = a in actual_top3_nums and b in actual_top3_nums
        combo = f"{a}-{b}"
        pay   = _get_payout(payouts, "ワイド", combo)
        results.append((combo, hit, pay))
    return results


def _check_sanrenpuku_raw(
    predicted_nums: list[int],
    actual_top3_nums: list[int],
    payouts: dict,
    ana_horse_num: Optional[int] = None,
    pred: Optional[dict] = None,
) -> tuple[bool, str]:
    """3連複的中判定。(hit, pay_str) を返す。

    predicted_top5_nums の全馬を相手候補として、
    軸(◎) × 相手2頭 の組合せが実際の3着以内と一致するか判定。
    """
    if len(actual_top3_nums) < 3:
        return False, ""

    # 軸 = honmei
    axis = None
    if pred:
        axis = (pred.get("honmei") or {}).get("horse_number")
    if axis is None and predicted_nums:
        axis = predicted_nums[0]
    if axis is None:
        return False, ""

    # 軸が3着以内に含まれることが前提
    if int(axis) not in actual_top3_nums:
        return False, ""

    # 相手候補: predicted_top5_nums を優先、なければ predicted_nums[1:5] + 穴馬
    partners = []
    if pred:
        top5 = pred.get("predicted_top5_nums", [])
        if top5:
            partners = [n for n in top5 if n != axis]
    if not partners:
        partners = list(predicted_nums[1:5]) if len(predicted_nums) > 1 else []
    if ana_horse_num and ana_horse_num not in partners:
        partners.append(ana_horse_num)

    # 軸 × 相手2頭 の全組合せで判定
    actual_set = set(actual_top3_nums[:3])
    for pair in combinations(partners, 2):
        if {int(axis), pair[0], pair[1]} == actual_set:
            combo = "-".join(str(n) for n in sorted([int(axis), pair[0], pair[1]]))
            pay = _get_payout(payouts, "三連複", combo)
            return True, pay
    return False, ""


def _format_prediction_from_cache(race_name: str, entry: dict, race_id: str = "") -> tuple[str, str]:
    """predictions_cache.json のエントリからDiscord用メッセージ(予想・買い目)を生成する。"""
    course_info = entry.get("course_info", "")
    ai_comments = entry.get("ai_comments", {})

    # ── Message 1: 予想 ───────────────────────────────────────
    venue = entry.get("venue", "")
    start_time = entry.get("start_time", "")
    race_num = ""
    if race_id and len(race_id) >= 12:
        try:
            race_num = f"{int(race_id[10:12])}R"
        except ValueError:
            pass
    flag_sep = "🏁━━━━━━━━━━━━━━━━━━🏁"
    title = f"　　{venue} {race_num} {race_name}".rstrip()
    meta_parts = []
    if venue:
        meta_parts.append(f"📍{venue}")
    if start_time:
        meta_parts.append(f"🕐{start_time}発走")
    if course_info:
        meta_parts.append(course_info)
    meta_line = f"　　{' | '.join(meta_parts)}" if meta_parts else ""
    lines1 = [flag_sep, title]
    if meta_line:
        lines1.append(meta_line)
    lines1.append(flag_sep)

    MARKS = ["◎", "○", "▲", "△", "　"]
    top5_nums = entry.get("predicted_top5_nums", [])

    # predicted_top5（上位5頭の詳細情報）を馬番→infoのマップに変換
    top5_detail: dict[int, dict] = {}
    for h in (entry.get("predicted_top5") or []):
        num = h.get("horse_number")
        if num is not None:
            top5_detail[int(num)] = h

    # honmei/taikou/ana からも補完
    for role in ("honmei", "taikou", "ana"):
        p = entry.get(role, {})
        num = p.get("horse_number")
        if num is not None and int(num) not in top5_detail:
            top5_detail[int(num)] = p

    ev_map: dict[int, dict] = {}
    for e in (entry.get("ev_top3") or []):
        num = e.get("horse_number")
        if num is not None:
            ev_map[int(num)] = e

    # モンテカルロ3着以内率マップ（あればXGBoostのprobより優先）
    sim = entry.get("simulation", {})

    for rank, num in enumerate(top5_nums):
        mark = MARKS[rank] if rank < len(MARKS) else "　"
        info = top5_detail.get(num, ev_map.get(num, {}))
        name = info.get("horse_name", "")
        if not name:
            name = f"{num}番"
        # MC確率を優先、なければ非表示（XGBoost生スコアは表示しない）
        mc_data = sim.get(str(num), {})
        mc_rate = mc_data.get("top3_rate")
        ev_entry = ev_map.get(num, {})
        ev_val = ev_entry.get("ev_score")
        has_real_odds = ev_entry.get("odds") is not None
        ev_str = f" EV{ev_val:.2f}" if ev_val and has_real_odds else ""
        prob_val = None
        if mc_rate is not None:
            prob_val = float(mc_rate)
        else:
            xgb_prob = info.get("prob")
            if xgb_prob is not None:
                try:
                    prob_val = float(xgb_prob)
                except (TypeError, ValueError):
                    prob_val = None
        if prob_val is not None:
            lines1.append(f"{mark} {num}番 {name}　{prob_val*100:.1f}%{ev_str}")
        else:
            lines1.append(f"{mark} {num}番 {name}{ev_str}")

    lines1.append(flag_sep)

    # ★穴馬
    ana_num = entry.get("ana_horse_num")
    ana_info = entry.get("ana_horse_info", {})
    if ana_num and ana_num not in top5_nums[:5]:
        name = ana_info.get("horse_name", "")
        pop = ana_info.get("popularity", "?")
        if not name:
            for e in entry.get("ev_top3", []):
                if e.get("horse_number") == ana_num:
                    name = e.get("horse_name", "")
                    break
        if name:
            # MC確率を優先
            mc_ana = sim.get(str(ana_num), {})
            mc_ana_rate = mc_ana.get("top3_rate")
            prob_val = None
            if mc_ana_rate is not None:
                prob_val = float(mc_ana_rate)
            else:
                xgb_prob = ana_info.get("prob")
                if xgb_prob is None:
                    for e in entry.get("ev_top3", []):
                        if e.get("horse_number") == ana_num:
                            xgb_prob = e.get("prob")
                            break
                if xgb_prob is not None:
                    try:
                        prob_val = float(xgb_prob)
                    except (TypeError, ValueError):
                        prob_val = None
            if prob_val is not None:
                lines1.append(f"★穴 {ana_num}番{name}（{prob_val*100:.1f}% {pop}番人気）")
            else:
                lines1.append(f"★穴 {ana_num}番{name}（{pop}番人気）")

    # ⚠危険馬
    for d in entry.get("dangerous_horses", []):
        num = d.get("horse_number", 0)
        name = d.get("horse_name", "")
        reasons = d.get("reasons", [])
        reason = reasons[0] if reasons else "要注意"
        lines1.append(f"⚠危険 {num}番{name}（{reason}）")

    msg1 = "\n".join(lines1)

    # ── Message 2: 買い目（bet_strategy があれば使用）──────────
    _SEP = "━" * 20
    bs = entry.get("bet_strategy")

    if bs and bs.get("total_points", 0) > 0:
        lines2 = ["💰 買い目"]

        # ワイド ◎-○ 1点
        if bs.get("wide"):
            w = bs["wide"][0]
            lines2.append(f"ワイド ◎{w['nums'][0]}-○{w['nums'][1]}  1,000円")

        lines2.append(f"────────────────")
        lines2.append(f"合計投資額: 1,000円")
    elif bs and "見送り" in str(bs.get("strategy_note", "")):
        # オッズフィルタで見送り
        note = bs.get("strategy_note", "見送り")
        lines2 = ["💰 買い目", f"⏭️ {note}"]
    else:
        # フォールバック: ワイド ◎-○ 1点（bet_strategyがない場合）
        nums = top5_nums
        if len(nums) < 2:
            return msg1, ""
        hon = nums[0]
        tai = nums[1]
        lines2 = [
            "💰 買い目",
            f"ワイド ◎{hon}-○{tai}  1,000円",
            f"────────────────",
            f"合計投資額: 1,000円",
        ]

    # 楽天競馬URL追加
    rakuten_url = _rakuten_race_url(race_id, entry.get("race_date", ""))
    if rakuten_url:
        lines2.append(f"🔗 {rakuten_url}")

    msg2 = "\n".join(lines2)
    return msg1, msg2


# ══════════════════════════════════════════════════════════════
# 機能1: 金曜予想
# ══════════════════════════════════════════════════════════════

def run_predict_notify(
    webhook_url: Optional[str] = None,
    featured_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    test_race_id: Optional[str] = None,
    use_live: bool = False,
) -> None:
    """週末重賞を予想して Discord に送信し、結果をキャッシュに保存する。

    Args:
        test_race_id: 指定時は週末重賞検索をスキップして該当race_idのみテスト送信する。
        use_live:     True のとき出馬表をリアルタイム取得して予測する。
                      出馬表未確定・取得失敗の場合は featured_races.csv にフォールバック。
    """
    webhook_url = _resolve_webhook(webhook_url)

    # 日付ベースの送信済みチェック（テスト時はスキップ）
    notified_flag = DATA_DIR / "notified_date.txt"
    if not test_race_id:
        today_str = _today_jst().isoformat()
        if notified_flag.exists():
            saved = notified_flag.read_text(encoding="utf-8").strip()
            if saved == today_str:
                logger.info(f"本日（{today_str}）は送信済み → スキップ")
                return

    if model_path is None:
        model_path = MODEL_PATH

    # 前提ファイル確認（モデルのみ必須、featured_races.csvはキャッシュ優先のため任意）
    if not model_path.exists():
        send_discord(webhook_url,
            "⚠️ モデルファイルが見つかりません。\n"
            "```\npython -m keiba_predictor.main train\n```")
        return

    model_bundle = load_model(model_path)

    # --test-race-id が指定された場合はレース検索をスキップ
    if test_race_id:
        race_name = str(test_race_id)
        grade_races = [{"race_id": test_race_id, "race_name": race_name, "race_date": "（テスト）"}]
        logger.info(f"テストモード: race_id={test_race_id} race_name={race_name}")
        send_discord(webhook_url, f"🧪 **テスト送信** race_id={test_race_id}  {race_name}")
    else:
        # NAR: 今日のレースを取得（重賞検索ではなく毎日開催）
        session = requests.Session()
        logger.info("今日のNARレースを検索中...")
        grade_races = scrape_nar_race_ids_for_today(session)
        if not grade_races:
            today_str = _today_jst().isoformat()
            send_discord(webhook_url, f"🐴 本日（{today_str}）のNARレースが見つかりませんでした。")
            return
        send_discord(webhook_url,
            f"🐴 **本日のNAR予想** ({_today_jst().isoformat()})  全{len(grade_races)}レース")

    notified = 0
    cache = _load_cache()

    # 昨日以前のレースをキャッシュから除外（NAR: 毎日開催のため当日分のみ保持）
    today_str = _today_jst().isoformat()
    stale_ids = [
        rid for rid, entry in cache.items()
        if not rid.startswith("_") and entry.get("race_date") and entry["race_date"] < today_str
    ]
    if stale_ids:
        for rid in stale_ids:
            del cache[rid]
        _save_cache(cache)
        logger.info(f"  {len(stale_ids)} 件の過去レースをキャッシュから削除")
    logger.info(f"本日: {today_str}")

    # 発走時刻順にソート（キャッシュの start_time を使用）
    grade_races = sorted(
        grade_races,
        key=lambda r: cache.get(r["race_id"], {}).get("start_time", "99:99"),
    )

    for race in grade_races:
        race_id   = race["race_id"]
        race_name = race.get("race_name", race_id)
        race_date = race.get("race_date", "")

        # キャッシュの race_date も確認
        cached_date = cache.get(race_id, {}).get("race_date", race_date)
        effective_date = cached_date or race_date

        if effective_date and effective_date != today_str:
            logger.info(f"  スキップ（{effective_date} ≠ {today_str}）: {race_name}")
            continue

        # 通知済みチェック（レース単位）
        if cache.get(race_id, {}).get("notified_predict"):
            logger.info(f"  通知済みスキップ: {race_name} ({race_id})")
            continue

        # ── キャッシュ優先: predictions_cache.json にデータがあればそれを使う ──
        # predict_live() を再実行するとcleaned_races.csvが無い環境で確率が壊れるため、
        # キャッシュに予想データがあれば常にそちらを使う
        cached_entry = cache.get(race_id, {})
        has_cache = bool(cached_entry and cached_entry.get("predicted_top3_nums"))

        if has_cache:
            race_name = cached_entry.get("race_name", race_name)
            logger.info(f"  キャッシュから予想を読み込み: {race_name} ({race_id})")

            # AI解説が空の場合は再生成を試みる
            if not cached_entry.get("ai_comments"):
                logger.info(f"  AI解説が空 → 再生成を試みます: {race_name}")
                try:
                    # キャッシュのpredicted_top5から簡易DataFrameを構築してAI解説生成
                    top5_data = cached_entry.get("predicted_top5", [])
                    if top5_data:
                        import pandas as _pd
                        ai_df = _pd.DataFrame(top5_data)
                        ai_df["prob_top3"] = ai_df["prob"]
                        if "ev_score" not in ai_df.columns:
                            ai_df["ev_score"] = ai_df.get("odds", _pd.Series(dtype=float)) * ai_df["prob"]
                        course_info = cached_entry.get("course_info", "")
                        ai_comments = generate_comments(
                            ai_df, race_name=race_name, course_info=course_info)
                        if ai_comments:
                            cached_entry["ai_comments"] = ai_comments
                            cache[race_id] = cached_entry
                            _save_cache(cache)
                            logger.info(f"  AI解説再生成成功: {len(ai_comments)} 頭分")
                except Exception as e:
                    logger.warning(f"  AI解説再生成失敗（続行）: {e}")

            msg1, msg2 = _format_prediction_from_cache(race_name, cached_entry, race_id=race_id)
        else:
            # キャッシュになければ predict_live() で生成
            logger.info(f"  キャッシュなし → predict_live 実行: {race_name} ({race_id})")
            try:
                from keiba_predictor.model.predict import predict_live
                result = predict_live(race_id, notify=False, model_path=model_path)
                cached_entry = _load_cache().get(race_id, {})
                ai_comments = cached_entry.get("ai_comments", {})
                course_info = cached_entry.get("course_info", "")
                race_name   = cached_entry.get("race_name", race_name)
                logger.info(f"  predict_live 成功: {race_name} ({race_id})")
                msg1, msg2 = format_prediction(result, race_name=race_name,
                                               ai_comments=ai_comments, course_info=course_info)
            except Exception as e:
                import traceback
                logger.warning(f"  predict_live 失敗: {traceback.format_exc()}")
                send_discord(webhook_url,
                    f"⚠️ **{race_name}** の予想生成に失敗しました: {e}")
                continue

        print(msg1, flush=True)
        print(msg2, flush=True)

        # Discord に送信（開催場別チャンネル振り分け）
        # 予想+買い目を1メッセージに結合（レース間が区別しやすい）
        venue = cached_entry.get("venue", "")
        target_url = _venue_webhook(venue, webhook_url)
        combined = msg1 + "\n\n" + msg2 if msg2 else msg1
        ok = send_discord(target_url, combined)
        if ok:
            notified += 1
            ch_label = venue if target_url != webhook_url else "default"
            logger.info(f"  送信完了: {race_name} → {ch_label}ch")
            # 通知済みフラグをキャッシュに保存
            if race_id in cache:
                cache[race_id]["notified_predict"] = True
                _save_cache(cache)

    send_discord(webhook_url, f"✅ {notified}/{len(grade_races)} レース送信完了")

    # X（Twitter）日次まとめ投稿（全会場の買い目を1ツイート/スレッドに集約）
    if os.environ.get("ENABLE_X_POST", "false").lower() == "true":
        try:
            from keiba_predictor.x_post import post_daily_bet_summary
            posted = post_daily_bet_summary(_load_cache())
            logger.info(f"[X] 買い目まとめ投稿: {posted}件")
        except Exception as e:
            logger.warning(f"[X] 買い目まとめ投稿エラー: {e}")

    # 送信済みフラグを書き込み
    if not test_race_id and notified > 0:
        try:
            notified_flag.write_text(_today_jst().isoformat(), encoding="utf-8")
            logger.info(f"送信済みフラグ書き込み: {notified_flag}")
        except Exception as e:
            logger.warning(f"送信済みフラグ書き込み失敗: {e}")


# ══════════════════════════════════════════════════════════════
# 機能2: 日曜結果
# ══════════════════════════════════════════════════════════════

def run_result_notify(
    webhook_url: Optional[str] = None,
    model_path: Optional[Path] = None,
    race_id: Optional[str] = None,
) -> None:
    """週末重賞の結果をスクレイピングし、予想との比較をDiscordに送信する。"""
    # 結果全般 → DISCORD_RESULT_WEBHOOK_URL（未設定ならDISCORD_WEBHOOK_URLにフォールバック）
    result_webhook = os.environ.get("DISCORD_RESULT_WEBHOOK_URL", "")
    if not result_webhook:
        result_webhook = _resolve_webhook(webhook_url)
    # 的中専用 → DISCORD_HIT_WEBHOOK_URL
    hit_webhook = os.environ.get("DISCORD_HIT_WEBHOOK_URL", "")
    # 従来互換（webhook_url引数）
    webhook_url = result_webhook

    session = requests.Session()
    cache   = _load_cache()

    # --race-id 指定時はそのレースのみ対象
    if race_id:
        cached = cache.get(race_id, {})
        race_name = cached.get("race_name", race_id)
        race_date = cached.get("race_date", "")
        grade_races = [{"race_id": race_id, "race_name": race_name, "race_date": race_date}]
        logger.info(f"指定レースID: {race_id} ({race_name})")
    else:
        # NAR: キャッシュ内の当日レースを結果照合対象にする
        logger.info("キャッシュから本日のNARレースを取得中...")
        today_str = _today_jst().isoformat()
        grade_races = []
        for rid, entry in cache.items():
            if rid.startswith("_"):
                continue
            if entry.get("race_date") == today_str:
                grade_races.append({
                    "race_id": rid,
                    "race_name": entry.get("race_name", rid),
                    "race_date": today_str,
                })
    if not grade_races:
        logger.info("本日のNARレースが見つかりませんでした。")
        return

    from keiba_predictor.scraper.netkeiba_scraper import scrape_nar_race_result as scrape_race_result
    from keiba_predictor.history import (
        record_result, load_history,
        weekly_summary, cumulative_summary, hit_streak, format_summary_message,
    )
    from datetime import date as _date

    # 手動結果を読み込む
    manual_results: dict = {}
    if MANUAL_RESULTS.exists():
        try:
            manual_results = json.loads(MANUAL_RESULTS.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"manual_results.json 読み込み失敗: {e}")

    # 既に通知済み（results_history.csv に記録済み）のrace_idをスキップ
    notified_ids: set[str] = set()
    try:
        hist_df = load_history()
        if not hist_df.empty:
            notified_ids = set(hist_df["race_id"].astype(str).unique())
            logger.info(f"通知済みレース: {len(notified_ids)} 件")
    except Exception:
        pass

    # 未通知レースがなければ何も送信せず終了
    pending = [r for r in grade_races if r["race_id"] not in notified_ids]
    if not pending:
        logger.info("全レース通知済み → スキップ")
        return

    notified = 0
    skipped  = 0
    for race in grade_races:
        race_id   = race["race_id"]
        race_name = race.get("race_name", race_id)
        race_date = race.get("race_date", "")

        # 既に結果通知済みならスキップ（2回目実行時の重複防止）
        if race_id in notified_ids:
            logger.info(f"  通知済みスキップ: {race_name} ({race_id})")
            skipped += 1
            continue

        # 手動結果があればスクレイピングをスキップ
        manual = manual_results.get(race_id)
        if manual:
            logger.info(f"  手動結果を使用: {race_id} ({manual.get('race_name', race_name)})")
            result_nums = manual.get("result", [])
            # 手動結果から簡易DataFrameを構築
            actual_rows = []
            for i, num in enumerate(result_nums):
                actual_rows.append({
                    "finish_position": i + 1,
                    "horse_number": num,
                    "horse_name": "",
                })
            actual_df = pd.DataFrame(actual_rows) if actual_rows else None
            # 払戻金
            manual_pay = manual.get("payouts", {})
            payouts = {}
            if manual_pay.get("umaren"):
                payouts["馬連"] = [{"combo": "-".join(str(n) for n in result_nums[:2]),
                                    "amount": manual_pay["umaren"]}]
            if manual_pay.get("sanrenpuku"):
                payouts["三連複"] = [{"combo": "-".join(str(n) for n in sorted(result_nums[:3])),
                                      "amount": manual_pay["sanrenpuku"]}]
        else:
            # 結果スクレイピング
            actual_df = scrape_race_result(race_id, session)
            payouts = scrape_payouts(race_id, session) if actual_df is not None else {}

        if actual_df is None or actual_df.empty:
            logger.info(f"  結果未確定スキップ: {race_name} ({race_id})")
            continue

        # 予想キャッシュ取得
        pred = cache.get(race_id, {})
        if not pred:
            logger.warning(f"  予想キャッシュなし: {race_id}")
            pred = {"race_name": race_name, "race_date": race_date,
                    "honmei": {}, "taikou": {}, "ana": {}, "predicted_top3_nums": []}

        # manual の honmei / predicted_top3_nums で pred を上書き
        if manual:
            if manual.get("honmei") is not None:
                pred["honmei"] = {"horse_number": manual["honmei"],
                                  "horse_name": pred.get("honmei", {}).get("horse_name", ""),
                                  "prob": pred.get("honmei", {}).get("prob", 0)}
            if manual.get("predicted_top3_nums"):
                pred["predicted_top3_nums"] = manual["predicted_top3_nums"]

        msg = _fmt_result(race_name, race_date, actual_df, pred, payouts, manual=manual, race_id=race_id)
        # 結果はデフォルトチャンネルに集約（予想は会場別、結果は一覧で振り返り）
        if send_discord(result_webhook, msg):
            notified += 1
            logger.info(f"  送信: {race_name} → 結果ch")

        # ── 的中時に専用チャンネルへ特別通知 ─────────────────────
        try:
            # 的中判定を再取得
            predicted_nums = pred.get("predicted_top3_nums", [])
            if manual and manual.get("predicted_top3_nums"):
                predicted_nums = manual["predicted_top3_nums"]

            if manual and manual.get("honmei") is not None:
                _honmei_num = int(manual["honmei"])
            elif predicted_nums:
                _honmei_num = predicted_nums[0]
            else:
                _honmei_num = (pred.get("honmei") or {}).get("horse_number")

            # 馬名マップ
            _num_to_name: dict[int, str] = {}
            for _role in ("honmei", "taikou", "ana"):
                _p = pred.get(_role, {})
                _pn = _p.get("horse_number")
                if _pn is not None:
                    _num_to_name[int(_pn)] = _p.get("horse_name", "")
            for _, _r in actual_df.iterrows():
                _an = _r.get("horse_number")
                _aname = str(_r.get("horse_name", ""))
                if pd.notna(_an) and _aname:
                    _num_to_name.setdefault(int(_an), _aname)
            _honmei_name = _num_to_name.get(_honmei_num, "") if _honmei_num else ""

            # 確定 1-3 着
            _df_copy = actual_df.copy()
            _df_copy["_fp"] = pd.to_numeric(_df_copy["finish_position"], errors="coerce")
            _top3 = _df_copy[_df_copy["_fp"].isin([1, 2, 3])].sort_values("_fp").head(3)
            _actual_top3_nums = [int(r["horse_number"]) for _, r in _top3.iterrows() if pd.notna(r.get("horse_number"))]

            # ワイド的中判定
            _wh = False
            _wp = ""
            _actual_set = set(_actual_top3_nums[:3])
            bs = pred.get("bet_strategy", {})
            if bs.get("wide"):
                for w in bs["wide"]:
                    a, b = w["nums"]
                    if a in _actual_set and b in _actual_set:
                        _wp = _get_payout(payouts, "ワイド", f"{a}-{b}")
                        _wh = True
                        break
            _venue = pred.get("venue", "")
            hit_embed = _build_hit_embed(
                _venue, race_name, _honmei_num, _honmei_name,
                _wh, _wp, race_id=race_id,
            )
            if hit_embed:
                target_webhook = hit_webhook if hit_webhook else result_webhook
                _send_hit_embed(target_webhook, hit_embed)
                label = "専用ch" if hit_webhook else "結果ch"
                logger.info(f"  的中GIF通知送信（{label}）: {race_name}")
        except Exception as e:
            logger.warning(f"  的中通知エラー ({race_name}): {e}")

        # 的中実績を CSV に記録
        if manual and "fukusho_hit" in manual:
            # manual_results.json の的中フラグで直接記録
            try:
                _record_manual_result(race_id, race_name, race_date, pred, manual)
            except Exception as e:
                logger.warning(f"  [history] 手動記録失敗 ({race_name}): {e}")
        else:
            try:
                record_result(race_id, race_name, race_date, pred, actual_df, payouts)
            except Exception as e:
                logger.warning(f"  [history] 記録失敗 ({race_name}): {e}")

        # X（Twitter）への個別レース結果投稿は廃止
        # → 日次まとめに集約（loss_analysis.py main で post_daily_result_summary を呼ぶ）

    if notified > 0:
        send_discord(webhook_url, f"✅ {notified}レース結果送信完了")

    # 日曜日に週次サマリーを X に投稿
    if os.environ.get("ENABLE_X_POST", "false").lower() == "true":
        try:
            today = _today_jst()
            if today.weekday() == 6:  # 日曜日
                from datetime import timedelta
                hist_df = load_history()
                week_start = today - timedelta(days=today.weekday())  # 月曜
                week_end = today
                ws = pd.Timestamp(week_start)
                we = pd.Timestamp(week_end)
                mask = (hist_df["date"] >= ws) & (
                    hist_df["date"] <= we + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
                wdf = hist_df[mask]
                if not wdf.empty:
                    results = []
                    for _, row in wdf.iterrows():
                        results.append({
                            "race_name": str(row.get("race_name", "")),
                            "fukusho": bool(row.get("fukusho_hit", False)),
                            "umaren": bool(row.get("umaren_hit", False)),
                            "sanren": bool(row.get("sanrenpuku_hit", False)),
                            "bet": int(row.get("bet_total", 0)),
                            "return_total": int(row.get("return_total", 0)),
                        })
                    from keiba_predictor.x_post import post_weekly_summary_tweet
                    post_weekly_summary_tweet(results)
        except Exception as e:
            logger.warning(f"  [X] 週次サマリー投稿エラー: {e}")


# ══════════════════════════════════════════════════════════════
# ユーティリティ
# ══════════════════════════════════════════════════════════════

def _resolve_webhook(url: Optional[str]) -> str:
    """引数 → 環境変数 → エラー の順で Webhook URL を解決する。"""
    if url:
        return url
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        raise ValueError(
            "Discord Webhook URL が未設定です。\n"
            "環境変数 DISCORD_WEBHOOK_URL を設定するか "
            "--webhook-url オプションを使用してください。"
        )
    return url


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="週末重賞 Discord 通知")
    p.add_argument("--mode", choices=["predict", "result"], required=True,
                   help="predict=金曜予想 / result=日曜結果")
    p.add_argument("--webhook-url", help="Discord Webhook URL（未指定=環境変数）")
    args = p.parse_args()

    if args.mode == "predict":
        run_predict_notify(webhook_url=args.webhook_url)
    else:
        run_result_notify(webhook_url=args.webhook_url)


if __name__ == "__main__":
    main()
