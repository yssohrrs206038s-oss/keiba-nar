"""
妥当性ガード (_validate_nar_result) のユニットテスト。

正常オッズ / 汚染オッズ（上がり3F値混入）/ overround超過 / popularity全NaN
の各ケースを検証する。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from keiba_predictor.scraper.netkeiba_scraper import _validate_nar_result


def _make_df(odds_list, pop_list=None):
    """テスト用 DataFrame を生成する。"""
    n = len(odds_list)
    if pop_list is None:
        pop_list = list(range(1, n + 1))
    return pd.DataFrame({
        "race_id":        ["test_race"] * n,
        "horse_number":   list(range(1, n + 1)),
        "finish_position": list(range(1, n + 1)),
        "horse_name":     [f"馬{i}" for i in range(1, n + 1)],
        "odds":           [str(o) if o is not None else "" for o in odds_list],
        "popularity":     [str(p) if p is not None else "" for p in pop_list],
    })


class TestValidateNarResult:

    def test_normal_odds_unchanged(self):
        """正常なオッズ（overround 1.1〜1.3）は odds/popularity が変更されない。"""
        # 8頭立て、各オッズを調整して overround≒1.2 にする
        odds = [2.5, 3.5, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
        df = _make_df(odds)
        result = _validate_nar_result(df.copy(), "test001")
        assert pd.to_numeric(result["odds"], errors="coerce").notna().all(), \
            "正常レースの odds が NaN になってはいけない"
        assert pd.to_numeric(result["popularity"], errors="coerce").notna().all(), \
            "正常レースの popularity が NaN になってはいけない"

    def test_corrupted_odds_last3f_values(self):
        """上がり3F値（35〜42秒台）が odds 列に混入した汚染ケース。"""
        # 上がり3F混入: overround = Σ(1/35〜42) ≒ 0.2 で異常
        last3f_values = [35.8, 36.2, 37.1, 38.5, 39.0, 39.8, 40.3, 41.2]
        df = _make_df(last3f_values, pop_list=[None] * 8)
        result = _validate_nar_result(df.copy(), "test002")
        assert pd.to_numeric(result["odds"], errors="coerce").isna().all(), \
            "汚染オッズ（上がり3F混入）は NaN に置換されるべき"
        assert pd.to_numeric(result["popularity"], errors="coerce").isna().all(), \
            "汚染オッズ時は popularity も NaN に置換されるべき"

    def test_overround_too_high(self):
        """overround が上限（1.8）を超えるケース。"""
        # 全馬オッズ 1.1 → overround ≒ 7.3（異常に高い）
        odds = [1.1] * 8
        df = _make_df(odds)
        result = _validate_nar_result(df.copy(), "test003")
        assert pd.to_numeric(result["odds"], errors="coerce").isna().all(), \
            "overround超過の odds は NaN に置換されるべき"

    def test_overround_borderline_lower(self):
        """overround が正常範囲（1.05〜1.8）内なら変更されない。"""
        # Σ(1/odds) ≈ 1.26 となるオッズ（実際の競馬レースに近い値）
        # 1/2.5+1/3.5+1/5.0+1/7.0+1/10.0+1/15.0+1/25.0+1/40.0 ≈ 1.261
        odds = [2.5, 3.5, 5.0, 7.0, 10.0, 15.0, 25.0, 40.0]
        df = _make_df(odds)
        result = _validate_nar_result(df.copy(), "test004")
        assert pd.to_numeric(result["odds"], errors="coerce").notna().all(), \
            "正常範囲のオッズは変更されない"

    def test_all_odds_nan_replaced(self):
        """odds が全て空文字（非数値）の場合も NaN 置換される。"""
        df = _make_df(["", "", "", ""])
        result = _validate_nar_result(df.copy(), "test005")
        assert pd.to_numeric(result["odds"], errors="coerce").isna().all()

    def test_popularity_all_nan_odds_kept(self):
        """popularity が全 NaN でも、odds が正常なら odds は保持される。"""
        odds = [2.5, 3.5, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
        df = _make_df(odds, pop_list=[None] * 8)
        result = _validate_nar_result(df.copy(), "test006")
        # odds は正常なので保持される
        assert pd.to_numeric(result["odds"], errors="coerce").notna().all(), \
            "popularity全NaN でも正常 odds は保持される"
        # popularity は NaN のまま（元々 NaN）
        assert pd.to_numeric(result["popularity"], errors="coerce").isna().all()

    def test_original_df_not_mutated(self):
        """元の DataFrame が副作用で変更されないことを確認。"""
        last3f_values = [35.8, 36.2, 37.1, 38.5, 39.0, 39.8, 40.3, 41.2]
        df = _make_df(last3f_values)
        original_odds = df["odds"].copy()
        _validate_nar_result(df, "test007")
        pd.testing.assert_series_equal(df["odds"], original_odds)


# ──────────────────────────────────────────────────────────────────
# ネットワーク確認テスト（実取得）
# ──────────────────────────────────────────────────────────────────

def test_live_scrape_header_and_overround():
    """
    直近のNAR race_id を1件実取得し、ヘッダーマッチ数とオーバーラウンドを確認する。
    ネットワーク不可の場合はスキップ。
    """
    import requests

    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            )
        })
        # ネットワーク疎通確認
        r = session.get("https://nar.netkeiba.com/", timeout=10)
        r.raise_for_status()
    except Exception as e:
        pytest.skip(f"ネットワーク不可のためスキップ: {e}")

    from keiba_predictor.scraper.netkeiba_scraper import (
        scrape_nar_kaisai_dates, scrape_nar_race_ids_for_date,
        scrape_nar_race_result,
    )
    from datetime import datetime, timedelta

    # 直近7日分の開催日からrace_idを探す
    today = datetime.today()
    race_id = None
    for delta in range(1, 8):
        dt = today - timedelta(days=delta)
        dates = scrape_nar_kaisai_dates(dt.year, dt.month, session)
        for d in dates:
            if d.startswith(f"{dt.year}{dt.month:02d}{dt.day:02d}"):
                ids = scrape_nar_race_ids_for_date(d, session)
                if ids:
                    race_id = ids[0]
                    break
        if race_id:
            break

    if race_id is None:
        pytest.skip("直近7日に開催が見つからなかったためスキップ")

    print(f"\n実取得対象: race_id={race_id}")
    df = scrape_nar_race_result(race_id, session)

    assert df is not None, f"scrape_nar_race_result が None を返した (race_id={race_id})"
    assert len(df) > 0, "結果が0行"

    odds_num = pd.to_numeric(df["odds"], errors="coerce")
    pop_num  = pd.to_numeric(df["popularity"], errors="coerce")

    valid_odds = odds_num.dropna()
    if len(valid_odds) > 0:
        overround = float((1.0 / valid_odds.clip(lower=0.01)).sum())
        print(f"  overround={overround:.3f}  (正常範囲: 1.05〜1.8)")
        print(f"  odds有効数={len(valid_odds)}/{len(df)}  popularity有効数={pop_num.notna().sum()}/{len(df)}")
        assert 1.05 <= overround <= 1.8, (
            f"オーバーラウンド異常: {overround:.3f}  "
            f"→ ヘッダーマッピングまたはエンコーディングを確認してください"
        )
    else:
        pytest.fail(f"全ての odds が非数値。race_id={race_id}")
