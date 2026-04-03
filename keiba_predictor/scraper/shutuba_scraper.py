"""
出馬表スクレイパー

netkeiba から出馬表を取得する。
  https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}
"""

import re
import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

from keiba_predictor.scraper.netkeiba_scraper import _get, HEADERS, VENUE_CODE_MAP

logger = logging.getLogger(__name__)

SHUTUBA_URL = "https://nar.netkeiba.com/race/shutuba.html"


class _PlaywrightSession:
    """Playwright ブラウザインスタンスを使い回すセッション管理。"""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._ctx = None
        self._page = None
        self._cookie_done = False

    def _ensure_browser(self):
        """ブラウザが起動していなければ起動する。"""
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._ctx = self._browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="ja-JP",
            extra_http_headers={
                "Accept-Language": HEADERS["Accept-Language"],
                "Accept": HEADERS["Accept"],
            },
        )
        self._page = self._ctx.new_page()
        logger.info("Playwright ブラウザセッション起動")

    def get_html(self, url: str) -> Optional[str]:
        """URL の HTML を取得して返す。失敗時は None。"""
        try:
            from playwright.sync_api import TimeoutError as PWTimeout
        except ImportError:
            logger.warning("playwright 未インストール → requests にフォールバック")
            return None

        try:
            self._ensure_browser()
            # 初回のみトップページで Cookie を取得
            if not self._cookie_done:
                self._page.goto(
                    "https://nar.netkeiba.com/",
                    wait_until="domcontentloaded", timeout=20000,
                )
                time.sleep(1)
                self._cookie_done = True
            self._page.goto(url, wait_until="domcontentloaded", timeout=60000)
            # 出馬表テーブルが描画されるまで最大10秒待機
            try:
                self._page.wait_for_selector(
                    "table.Shutuba_Table, tr.HorseList, td.Umaban",
                    timeout=10000,
                )
            except PWTimeout:
                logger.warning("Playwright: 出馬表セレクタのタイムアウト（HTML をそのまま使用）")
            html = self._page.content()
            # Playwright はブラウザ経由のため通常 UTF-8 変換済み。
            # 万が一 EUC-JP バイトが混在している場合のフォールバック
            try:
                html.encode("utf-8")
            except UnicodeEncodeError:
                try:
                    html = html.encode("latin-1").decode("euc-jp", errors="replace")
                    logger.info("Playwright HTML を EUC-JP → UTF-8 に再変換しました")
                except Exception:
                    pass
            logger.info(f"Playwright 取得成功: {len(html)} bytes")
            return html
        except Exception as e:
            logger.warning(f"Playwright 取得失敗: {e}")
            self.close()
            return None

    def close(self):
        """ブラウザセッションを明示的に閉じる。"""
        try:
            if self._browser:
                self._browser.close()
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._browser = None
        self._pw = None
        self._ctx = None
        self._page = None
        self._cookie_done = False


# モジュールレベルのシングルトン（複数レース処理でブラウザを使い回す）
_pw_session = _PlaywrightSession()


def _get_html_with_playwright(url: str) -> Optional[str]:
    """
    Playwright (Chromium) で JS レンダリング後の HTML を返す。
    playwright 未インストール時は None を返す。
    ブラウザインスタンスはモジュール内で使い回す。
    """
    return _pw_session.get_html(url)

# 性別エンコード（data_cleaner.py と合わせる）
_SEX_ENC = {"牡": 0, "牝": 1, "セ": 2, "騸": 2}


def _is_cancel_row(tr) -> bool:
    """tr が取消馬かどうかを判定する。"""
    tr_classes = tr.get("class", [])
    if "Cancel" in tr_classes:
        return True
    for td in tr.find_all("td"):
        td_cls = td.get("class") or []
        if isinstance(td_cls, list):
            td_cls_str = " ".join(td_cls)
        else:
            td_cls_str = str(td_cls)
        if "Cancel" in td_cls_str:
            return True
        if td.get_text(strip=True) == "取消":
            return True
    return False


def _parse_horse_weight(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    "486(+2)"  → (486.0,  2.0)
    "486(-4)"  → (486.0, -4.0)
    "486"      → (486.0, None)
    """
    if not isinstance(s, str):
        return None, None
    s = s.strip()
    m = re.match(r"(\d+)\s*\(([+-]?\d+)\)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), None
    except ValueError:
        return None, None


def _parse_sex_age(s: str) -> tuple[Optional[str], Optional[int]]:
    """"牡3" → ("牡", 3)"""
    if not isinstance(s, str):
        return None, None
    m = re.match(r"([牡牝セ騸])(\d+)", s.strip())
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def _parse_shutuba_row(tr) -> Optional[dict]:
    """<tr class="HorseList"> 1行から馬情報を抽出する。取消馬は None を返す。"""
    # 取消馬をスキップ
    if _is_cancel_row(tr):
        _horse_el = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
        _name = _horse_el.get_text(strip=True) if _horse_el else "?"
        print(f"[CANCEL SKIP] {_name}: tr.class={tr.get('class', [])}", flush=True)
        logger.info(f"取消馬スキップ: {_name}")
        return None

    def _txt(*sels):
        for sel in sels:
            el = tr.select_one(sel)
            if el:
                return el.get_text(strip=True)
        return ""

    def _link_id(sel, pattern):
        el = tr.select_one(sel)
        if el and el.get("href"):
            m = re.search(pattern, el["href"])
            return m.group(1) if m else ""
        return ""

    # 馬番（必須）
    try:
        horse_number = int(_txt("td[class*='Umaban']", ".Umaban", "td.Umaban"))
    except ValueError:
        return None

    # 枠番
    try:
        frame_number = int(_txt("td[class*='Waku']", ".Waku", "td.Waku"))
    except ValueError:
        frame_number = (horse_number - 1) // 2 + 1

    # 馬名・馬ID
    horse_link = tr.select_one(".HorseName a") or tr.select_one("td.HorseInfo a")
    horse_name = horse_link.get_text(strip=True) if horse_link else ""
    horse_id = ""
    if horse_link and horse_link.get("href"):
        m = re.search(r"/horse/(\w+)/?", horse_link["href"])
        horse_id = m.group(1) if m else ""

    # 性齢（NAR: classなしのtd[4]にフォールバック）
    sex_age_text = _txt(".Barei", "td.Barei", "td.sexage")
    if not sex_age_text:
        tds = tr.select("td")
        if len(tds) > 4:
            sex_age_text = tds[4].get_text(strip=True)
    sex, age = _parse_sex_age(sex_age_text)
    sex_enc = _SEX_ENC.get(sex, 0) if sex else 0

    # 斤量（NAR: .Txt_C または td[5]にフォールバック）
    try:
        wc_text = _txt(".Futan", "td.Futan", "td.Wt", ".Txt_C")
        if not wc_text:
            tds = tr.select("td")
            if len(tds) > 5:
                wc_text = tds[5].get_text(strip=True)
        weight_carried = float(wc_text) if wc_text else None
    except ValueError:
        weight_carried = None

    # 馬体重（NAR: td.Weight）
    horse_weight, horse_weight_diff = _parse_horse_weight(
        _txt(".HorseWeight", "td.HorseWeight", "td.Weight")
    )

    # 騎手・騎手ID
    jockey_link = tr.select_one(".Jockey a") or tr.select_one("td.Jockey a")
    jockey = jockey_link.get_text(strip=True) if jockey_link else ""
    jockey_id = ""
    if jockey_link and jockey_link.get("href"):
        m = re.search(r"/jockey/(?:result/recent/)?(\w+)/?", jockey_link["href"])
        jockey_id = m.group(1) if m else ""

    # 調教師・調教師ID
    trainer_link = tr.select_one(".Trainer a") or tr.select_one("td.Trainer a")
    trainer = trainer_link.get_text(strip=True) if trainer_link else ""
    trainer_id = ""
    if trainer_link and trainer_link.get("href"):
        m = re.search(r"/trainer/(?:result/recent/)?(\w+)/?", trainer_link["href"])
        trainer_id = m.group(1) if m else ""

    # オッズ（発走前は "---" の場合あり）
    # netkeibaの出馬表では複数のHTML構造が存在する
    odds = None
    for odds_sel in (".Odds span", ".Odds", "td.Odds span", "td.Odds",
                     "td.Txt_R", "span.Odds", ".OddsList span"):
        el = tr.select_one(odds_sel)
        if el:
            odds_text = el.get_text(strip=True).replace(",", "").replace("---", "")
            if odds_text:
                try:
                    odds = float(odds_text)
                    break
                except ValueError:
                    continue
    # フォールバック: tdを走査してオッズらしい数値を探す
    if odds is None:
        for td in tr.find_all("td"):
            td_text = td.get_text(strip=True).replace(",", "")
            td_cls = " ".join(td.get("class") or [])
            # Oddsを含むクラス、またはPopular系クラスの数値
            if re.match(r"^\d+\.\d+$", td_text) and ("Odds" in td_cls or "Txt_R" in td_cls or "Popular" in td_cls):
                try:
                    odds = float(td_text)
                    break
                except ValueError:
                    continue

    # 人気（NAR: td.Popular.Txt_C にフォールバック）
    popularity = None
    for pop_sel in (".Popular_Ninki", "td.Popular_Ninki", ".Popular span",
                    "td.Popular.Txt_C", "td.Popular", "td.popular_rank",
                    "span.Popular_Ninki"):
        el = tr.select_one(pop_sel)
        if el:
            pop_text = el.get_text(strip=True)
            if pop_text.isdigit():
                popularity = int(pop_text)
                break
    # インデックスフォールバック: td[10]
    if popularity is None:
        tds = tr.select("td")
        if len(tds) > 10:
            pop_text = tds[10].get_text(strip=True)
            if pop_text.isdigit():
                popularity = int(pop_text)

    # 脚質（逃/先/差/追）
    _RUNNING_STYLE_MAP = {"逃": 0, "先": 1, "差": 2, "追": 3, "逃げ": 0, "先行": 1, "差し": 2, "追込": 3, "追い込み": 3}
    running_style_enc = None
    for rs_sel in (".RunningStyle", "td.RunningStyle", ".Style", "td.Style",
                   "span.RunningStyle", ".HorseInfo span"):
        el = tr.select_one(rs_sel)
        if el:
            rs_text = el.get_text(strip=True)
            if rs_text in _RUNNING_STYLE_MAP:
                running_style_enc = _RUNNING_STYLE_MAP[rs_text]
                break
    # フォールバック: 通過順位から推定（1角1〜2番手→逃/先）
    if running_style_enc is None:
        pass_text = _txt("td.PassageRate", ".PassageRate")
        if pass_text:
            first_pos = re.match(r"(\d+)", pass_text)
            if first_pos:
                pos = int(first_pos.group(1))
                if pos <= 2:
                    running_style_enc = 0  # 逃
                elif pos <= 5:
                    running_style_enc = 1  # 先
                elif pos <= 10:
                    running_style_enc = 2  # 差
                else:
                    running_style_enc = 3  # 追

    return {
        "horse_number":       horse_number,
        "frame_number":       frame_number,
        "horse_name":         horse_name,
        "horse_id":           horse_id,
        "sex":                sex or "",
        "sex_enc":            sex_enc,
        "age":                age,
        "weight_carried":     weight_carried,
        "horse_weight":       horse_weight,
        "horse_weight_diff":  horse_weight_diff,
        "jockey":             jockey,
        "jockey_id":          jockey_id,
        "trainer":            trainer,
        "trainer_id":         trainer_id,
        "odds":               odds,
        "popularity":         popularity,
        "running_style_enc":  running_style_enc,
    }


def scrape_shutuba(race_id: str) -> Optional[dict]:
    """
    netkeiba の出馬表ページから馬情報とレース基本情報を取得する。

    Returns:
        {
            "race_id":         str,
            "race_name":       str,
            "race_date":       str,       # "YYYY-MM-DD"
            "venue":           str,
            "course_info":     str,       # "芝1800m"
            "distance":        int,
            "course_type_enc": int,       # 1=芝 / 0=ダート
            "race_grade_enc":  int,
            "horses":          pd.DataFrame,
        }
        取得失敗時は None。
    """
    url = f"{SHUTUBA_URL}?race_id={race_id}"
    logger.info(f"出馬表を取得: {url}")

    # ── 取得方法1: Playwright ────────────────────────────────
    soup = None
    html = _get_html_with_playwright(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        logger.info("Playwright で HTML 取得成功")

    # ── 取得方法2: requests ──────────────────────────────────
    if soup is None:
        logger.info("requests にフォールバック")
        session = requests.Session()
        session.headers.update(HEADERS)
        try:
            session.get("https://nar.netkeiba.com/", headers=HEADERS, timeout=15)
            time.sleep(random.uniform(0.5, 1.5))
        except Exception:
            pass
        soup = _get(url, session, encoding="euc-jp")

    if soup is None:
        logger.error(f"出馬表の取得に失敗: {race_id}")
        return None

    # ── デバッグ: 取得HTMLをファイルに保存 ─────────────────────
    try:
        import pathlib
        _debug_dir = pathlib.Path(__file__).parent.parent / "data" / "debug"
        _debug_dir.mkdir(parents=True, exist_ok=True)
        _debug_path = _debug_dir / f"shutuba_{race_id}.html"
        _debug_path.write_text(soup.prettify(), encoding="utf-8")
        logger.info(f"HTML保存: {_debug_path.name}")
    except Exception:
        pass

    # ── レース基本情報 ─────────────────────────────────────
    race_name = ""
    for sel in (".RaceName", "h1.RaceName", ".RaceTitle"):
        el = soup.select_one(sel)
        if el:
            race_name = el.get_text(strip=True)
            break

    # 日付
    race_date = ""
    for sel in (".RaceData01", ".Race_Date", ".RaceInfo", ".RaceList_DataItem"):
        el = soup.select_one(sel)
        if el:
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", el.get_text())
            if m:
                race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                break
    if not race_date:
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", soup.get_text())
        if m:
            race_date = f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if race_date:
        logger.info(f"開催日をHTMLから取得: {race_date}")
    else:
        logger.warning(f"開催日をHTMLから取得できませんでした (race_id={race_id})")

    # 発走時間
    start_time = ""
    mt = re.search(r"(\d{1,2}):(\d{2})発走", soup.get_text())
    if mt:
        start_time = f"{int(mt.group(1)):02d}:{mt.group(2)}"

    # 会場
    venue = VENUE_CODE_MAP.get(str(race_id)[4:6], "")

    # コース・距離
    distance = 0
    course_type_enc = 1  # デフォルト芝
    course_info = ""
    track_condition_enc = None  # 良=0, 稍重=1, 重=2, 不良=3
    _TRACK_COND_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
    data01 = soup.select_one(".RaceData01") or soup.select_one(".RaceInfo")
    if data01:
        txt = data01.get_text()
        m = re.search(r"(\d{3,4})m", txt)
        if m:
            distance = int(m.group(1))
        if "ダート" in txt or re.search(r"\bダ\b", txt):
            course_type_enc = 0
            course_info = f"ダート{distance}m"
        else:
            course_type_enc = 1
            course_info = f"芝{distance}m"
        # 馬場状態
        for cond, enc in _TRACK_COND_MAP.items():
            if cond in txt:
                track_condition_enc = enc
                break
    # RaceData02 からも馬場状態を探す
    if track_condition_enc is None:
        data02 = soup.select_one(".RaceData02") or soup.select_one(".Race_Data")
        if data02:
            txt2 = data02.get_text()
            for cond, enc in _TRACK_COND_MAP.items():
                if cond in txt2:
                    track_condition_enc = enc
                    break
    if track_condition_enc is not None:
        logger.info(f"馬場状態: {list(_TRACK_COND_MAP.keys())[track_condition_enc]} (enc={track_condition_enc})")

    # レース格
    from keiba_predictor.features.feature_engineering import _encode_race_grade
    race_grade_enc = _encode_race_grade(race_name)

    # ── 出馬表テーブル ──────────────────────────────────────
    table = (
        soup.select_one("table.Shutuba_Table")
        or soup.select_one("table#shutuba_table")
        or soup.select_one("table.ShutubaTable")
        or soup.select_one("table[class*='Shutuba']")
    )

    rows = []
    if table:
        trs = table.select("tr.HorseList, tr[class*='HorseList']")
        if not trs:
            trs = [tr for tr in table.find_all("tr")
                   if tr.select_one("td[class*='Umaban']") or tr.select_one("td.Umaban")]
        logger.info(f"HorseList 行数: {len(trs)}")
        for tr in trs:
            tr_classes = tr.get("class", [])
            _hel = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
            _hn = _hel.get_text(strip=True) if _hel else "?"
            # 取消馬スキップ
            if "Cancel" in tr_classes:
                logger.info(f"取消馬スキップ: {_hn}")
                continue
            if any("Cancel" in str(td.get("class", "")) for td in tr.find_all("td")):
                logger.info(f"取消馬スキップ: {_hn}")
                continue
            if any(td.get_text(strip=True) == "取消" for td in tr.find_all("td")):
                logger.info(f"取消馬スキップ: {_hn}")
                continue
            row = _parse_shutuba_row(tr)
            if row:
                rows.append(row)
    else:
        logger.warning("出馬表テーブルが見つかりませんでした（selector: table.Shutuba_Table）")
        horse_trs = soup.select("tr.HorseList, tr[class*='HorseList']")
        logger.info(f"ページ全体の HorseList 行数: {len(horse_trs)}")
        for tr in horse_trs:
            tr_classes = tr.get("class", [])
            _hel = tr.select_one(".HorseName a") or tr.select_one(".HorseInfo a")
            _hn = _hel.get_text(strip=True) if _hel else "?"
            # 取消馬スキップ
            if "Cancel" in tr_classes:
                logger.info(f"取消馬スキップ: {_hn}")
                continue
            if any("Cancel" in str(td.get("class", "")) for td in tr.find_all("td")):
                logger.info(f"取消馬スキップ: {_hn}")
                continue
            if any(td.get_text(strip=True) == "取消" for td in tr.find_all("td")):
                logger.info(f"取消馬スキップ: {_hn}")
                continue
            row = _parse_shutuba_row(tr)
            if row:
                rows.append(row)

    if not rows:
        logger.warning(f"出馬表の行データが 0 件です（race_id={race_id}）")

    horses_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    logger.info(
        f"出馬表取得完了: {race_name} {race_date} {start_time} {course_info} / {len(horses_df)}頭"
    )
    return {
        "race_id":         race_id,
        "race_name":       race_name,
        "race_date":       race_date,
        "start_time":      start_time,
        "venue":           venue,
        "course_info":     course_info,
        "distance":        distance,
        "course_type_enc":     course_type_enc,
        "track_condition_enc": track_condition_enc,
        "race_grade_enc":      race_grade_enc,
        "horses":              horses_df,
    }
