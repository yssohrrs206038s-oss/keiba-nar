"""
新特徴量（クラス昇降・Elo・騎手乗り替わり）のユニットテスト。
ネットワーク不要。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from keiba_predictor.features.feature_engineering import (
    parse_race_class,
    build_elo_table,
    add_jockey_change_feature,
    add_race_class_features,
)


# ── parse_race_class ──────────────────────────────────────────────

class TestParseRaceClass:
    def test_c3(self):
        assert parse_race_class("C3三") == 1.0

    def test_mixed_b2b3_takes_upper(self):
        assert parse_race_class("B2B3") == 5.0  # B2=5 > B3=4

    def test_a1_with_parens(self):
        assert parse_race_class("A1(A1A2)") == 9.0

    def test_race_name_with_prefix(self):
        assert parse_race_class("ダリア賞B1B2") == 6.0  # B1=6 > B2=5

    def test_grade_race(self):
        assert parse_race_class("帝王賞（重賞）") == 10.0

    def test_undeterminable(self):
        assert np.isnan(parse_race_class("第1回フリーカップ"))

    def test_none_input(self):
        assert np.isnan(parse_race_class(None))

    def test_empty_string(self):
        assert np.isnan(parse_race_class(""))

    def test_shinba(self):
        assert parse_race_class("新馬戦") == 0.0

    def test_open(self):
        assert parse_race_class("南関東OP") == 10.0


# ── build_elo_table ───────────────────────────────────────────────

def _make_elo_df():
    """
    3レース×3頭の小テストデータ。
    race1: 馬A(1着) > 馬B(2着) > 馬C(3着)  全員初戦 → elo=1500
    race2: 馬A(2着) < 馬B(1着)              horse_id なし馬Dも出走（対象外）
    race3: 馬C(1着) > 馬A(2着)
    """
    rows = [
        # race1
        {"race_id": "r1", "race_date": "2024-01-01", "horse_id": "A", "finish_position": 1},
        {"race_id": "r1", "race_date": "2024-01-01", "horse_id": "B", "finish_position": 2},
        {"race_id": "r1", "race_date": "2024-01-01", "horse_id": "C", "finish_position": 3},
        # race2
        {"race_id": "r2", "race_date": "2024-02-01", "horse_id": "A", "finish_position": 2},
        {"race_id": "r2", "race_date": "2024-02-01", "horse_id": "B", "finish_position": 1},
        {"race_id": "r2", "race_date": "2024-02-01", "horse_id": None, "finish_position": 3},  # horse_id なし
        # race3
        {"race_id": "r3", "race_date": "2024-03-01", "horse_id": "C", "finish_position": 1},
        {"race_id": "r3", "race_date": "2024-03-01", "horse_id": "A", "finish_position": 2},
    ]
    return pd.DataFrame(rows)


class TestBuildEloTable:
    def setup_method(self):
        self.df = _make_elo_df()
        self.elo = build_elo_table(self.df)

    def test_first_race_elo_is_init(self):
        """初戦の馬は Elo = 1500 であること。"""
        r1_mask = self.df["race_id"] == "r1"
        r1_elos = self.elo.loc[r1_mask, "horse_elo"]
        assert (r1_elos == 1500.0).all()

    def test_no_horse_id_is_nan(self):
        """horse_id が None の馬は horse_elo が NaN であること。"""
        none_mask = self.df["horse_id"].isna()
        assert self.elo.loc[none_mask, "horse_elo"].isna().all()

    def test_winner_elo_increases_after_race1(self):
        """race1 で1着の馬Aは race2 で Elo > 1500 になっている。"""
        r2_a = self.df[(self.df["race_id"] == "r2") & (self.df["horse_id"] == "A")].index
        assert float(self.elo.loc[r2_a[0], "horse_elo"]) > 1500.0

    def test_loser_elo_decreases_after_race1(self):
        """race1 で3着の馬Cは race2 が Elo < 1500（ただし race2 未出走なので race3 で確認）。"""
        r3_c = self.df[(self.df["race_id"] == "r3") & (self.df["horse_id"] == "C")].index
        # C は race1 で3着 → Elo 下がる → race3 時点で 1500 未満
        assert float(self.elo.loc[r3_c[0], "horse_elo"]) < 1500.0

    def test_pre_race_elo_recorded(self):
        """全有効行に horse_elo が記録されていること。"""
        valid_mask = self.df["horse_id"].notna()
        assert self.elo.loc[valid_mask, "horse_elo"].notna().all()


# ── add_jockey_change_feature ─────────────────────────────────────

class TestJockeyChange:
    def test_change_detected(self):
        df = pd.DataFrame({
            "horse_id": ["H1", "H1", "H1"],
            "race_id":  ["r1", "r2", "r3"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "jockey_id": ["J1", "J2", "J2"],
        })
        result = add_jockey_change_feature(df.copy())
        vals = result.set_index("race_date")["is_jockey_change"]
        assert np.isnan(vals["2024-01-01"])           # 初戦は NaN
        assert vals["2024-02-01"] == 1.0              # 変更あり
        assert vals["2024-03-01"] == 0.0              # 変更なし

    def test_no_jockey_id_column(self):
        df = pd.DataFrame({
            "horse_id": ["H1", "H1"],
            "race_id":  ["r1", "r2"],
            "race_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
        })
        result = add_jockey_change_feature(df.copy())
        assert result["is_jockey_change"].isna().all()
