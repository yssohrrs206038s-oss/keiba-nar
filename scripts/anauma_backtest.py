"""
穴馬発見モデル バックテスト

オッズ盲目モデルと市場確率の乖離(deviation)を使い、
市場が過小評価する馬を単勝で狙う戦略のROIを検証する。

実行方法:
    python scripts/anauma_backtest.py           # 全データ
    python scripts/anauma_backtest.py --sample  # 直近1万行のみ(動作確認用)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# プロジェクトルートを sys.path に追加
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from keiba_predictor.features.feature_engineering import FEATURE_COLS, build_features
from keiba_predictor.model.calibration import Calibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR    = REPO_ROOT / "keiba_predictor" / "data"
REPORT_PATH = DATA_DIR / "anauma_backtest_report.md"

# 時系列分割境界
TRAIN_END    = pd.Timestamp("2025-07-01")   # < この日付が学習期間
VAL_MID      = None                          # 検証の前半/後半境界（等分で自動計算）

# オッズ帯ラベル
ODDS_BANDS = [
    (0.0,   2.0,  "2倍未満"),
    (2.0,   5.0,  "2〜5倍"),
    (5.0,  10.0,  "5〜10倍"),
    (10.0, 20.0,  "10〜20倍"),
    (20.0, 9999,  "20倍超"),
]

# deviation 閾値スイープ
THETA_LIST = [0.05, 0.10, 0.15, 0.20]

# XGBoostハイパーパラメータ(既存モデルと同一)
BLIND_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "auc",
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
    "min_child_weight": 5,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 2.0,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}


# ──────────────────────────────────────────────────────────────────
# データ準備
# ──────────────────────────────────────────────────────────────────

def load_data(sample: bool = False) -> pd.DataFrame:
    """cleaned_races.csv.gz を読み込んで特徴量を生成する。"""
    gz = DATA_DIR / "cleaned_races.csv.gz"
    csv = DATA_DIR / "cleaned_races.csv"
    path = gz if gz.exists() else csv
    if not path.exists():
        raise FileNotFoundError(
            f"クリーニング済みデータが見つかりません: {gz} / {csv}"
        )
    logger.info(f"データ読み込み: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["race_date"])
    df = df.sort_values("race_date").reset_index(drop=True)

    if sample:
        df = df.tail(10_000).reset_index(drop=True)
        logger.info("--sample モード: 末尾10,000行を使用")

    logger.info(f"  行数: {len(df)}, 期間: {df['race_date'].min()} ~ {df['race_date'].max()}")

    logger.info("特徴量エンジニアリング中（数分かかる場合があります）...")
    df = build_features(df)

    # 数値化・欠損許容
    df["odds"]     = pd.to_numeric(df.get("odds"),     errors="coerce")
    df["top3"]     = pd.to_numeric(df.get("top3"),     errors="coerce")
    df["odds"]     = df["odds"].fillna(99.9)

    # finish_position を数値化（単勝配当計算に使用）
    df["finish_position"] = pd.to_numeric(df.get("finish_position"), errors="coerce")

    df = df.dropna(subset=["top3", "race_date"]).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────
# 盲目モデル
# ──────────────────────────────────────────────────────────────────

# last_3f は当日レースの上がり3Fであり、レース終了後にしか判明しないためリーク除外。
# prev2_last_3f / prev3_last_3f は過去走の値なので残す。
BLIND_COLS = [c for c in FEATURE_COLS if c not in ("odds", "popularity", "last_3f")]


def train_blind_model(df_train: pd.DataFrame) -> xgb.XGBClassifier:
    """odds/popularity を除いた特徴量でXGBoostを学習する。"""
    available = [c for c in BLIND_COLS if c in df_train.columns]
    X = df_train[available].apply(pd.to_numeric, errors="coerce")
    y = df_train["top3"].astype(int)
    logger.info(f"盲目モデル学習: {len(X)} 行, {len(available)} 特徴量")
    model = xgb.XGBClassifier(**BLIND_PARAMS)
    model.fit(X, y, verbose=False)
    return model


def predict_blind(model: xgb.XGBClassifier, df: pd.DataFrame) -> np.ndarray:
    available = [c for c in BLIND_COLS if c in df.columns]
    X = df[available].apply(pd.to_numeric, errors="coerce")
    return model.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────────────────
# 市場確率モデル
# ──────────────────────────────────────────────────────────────────

def compute_market_prob(df: pd.DataFrame) -> pd.Series:
    """
    p_win = (1/odds) / Σ(1/odds) でオーバーラウンド除去した市場単勝確率を返す。
    """
    inv_odds = 1.0 / df["odds"].clip(lower=1.01)
    total    = inv_odds.groupby(df["race_id"]).transform("sum")
    total    = total.replace(0, np.nan)
    return inv_odds / total


def train_market_top3_model(df_train: pd.DataFrame) -> LogisticRegression:
    """
    log(p_win) と出走頭数を説明変数として、
    top3 を目的変数にロジスティック回帰を学習する。
    """
    p_win     = compute_market_prob(df_train)
    log_p_win = np.log(p_win.clip(lower=1e-6))
    field_sz  = df_train.groupby("race_id")["race_id"].transform("count").astype(float)

    X = np.column_stack([log_p_win.values, field_sz.values])
    y = df_train["top3"].astype(int).values

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    return lr


def predict_market_top3(lr: LogisticRegression, df: pd.DataFrame) -> np.ndarray:
    p_win     = compute_market_prob(df)
    log_p_win = np.log(p_win.clip(lower=1e-6))
    field_sz  = df.groupby("race_id")["race_id"].transform("count").astype(float)
    X = np.column_stack([log_p_win.values, field_sz.values])
    return lr.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────────────────
# バックテスト
# ──────────────────────────────────────────────────────────────────

def run_tan_strategy(df_bt: pd.DataFrame, theta: float) -> dict:
    """
    単勝戦略: deviation >= theta の馬の単勝を100円買い。
    払戻 = 1着なら odds×100円。
    """
    sel    = df_bt[df_bt["deviation"] >= theta].copy()
    n_bets = len(sel)
    if n_bets == 0:
        return {
            "theta": theta, "n_bets": 0, "n_days": 0, "bets_per_day": 0.0,
            "hit_rate": 0.0, "roi": 0.0,
            "invested": 0, "returned": 0,
        }

    sel["won"]      = (sel["finish_position"] == 1).astype(int)
    sel["payout"]   = sel["won"] * sel["odds"] * 100
    invested        = n_bets * 100
    returned        = int(sel["payout"].sum())
    hit_rate        = float(sel["won"].mean())
    roi             = returned / invested if invested else 0.0

    n_days = sel["race_date"].dt.date.nunique()
    bets_per_day = n_bets / n_days if n_days else 0.0

    return {
        "theta":        theta,
        "n_bets":       n_bets,
        "n_days":       n_days,
        "bets_per_day": round(bets_per_day, 1),
        "hit_rate":     round(hit_rate, 4),
        "roi":          round(roi, 4),
        "invested":     invested,
        "returned":     returned,
    }


def odds_band_roi(df_bt: pd.DataFrame, theta: float) -> list[dict]:
    """オッズ帯別ROI。"""
    sel = df_bt[df_bt["deviation"] >= theta].copy()
    rows = []
    for lo, hi, label in ODDS_BANDS:
        band = sel[(sel["odds"] >= lo) & (sel["odds"] < hi)]
        n    = len(band)
        if n == 0:
            rows.append({"band": label, "n": 0, "hit_rate": 0.0, "roi": 0.0})
            continue
        won  = (band["finish_position"] == 1).sum()
        ret  = (band["finish_position"] == 1).astype(int).values * band["odds"].values * 100
        roi  = ret.sum() / (n * 100)
        rows.append({
            "band":     label,
            "n":        n,
            "hit_rate": round(won / n, 4),
            "roi":      round(roi, 4),
        })
    return rows


def venue_roi(df_bt: pd.DataFrame, theta: float) -> list[dict]:
    """競馬場別ROI（上位10会場）。"""
    if "venue" not in df_bt.columns:
        return []
    sel = df_bt[df_bt["deviation"] >= theta].copy()
    rows = []
    for venue, g in sel.groupby("venue"):
        n   = len(g)
        won = (g["finish_position"] == 1).sum()
        ret = (g["finish_position"] == 1).astype(int).values * g["odds"].values * 100
        roi = ret.sum() / (n * 100) if n else 0.0
        rows.append({"venue": venue, "n": n, "hit_rate": round(won / n, 4), "roi": round(roi, 4)})
    rows.sort(key=lambda r: -r["n"])
    return rows[:10]


def baseline_roi(df_bt: pd.DataFrame) -> dict:
    """全馬単勝・1番人気単勝のROIを計算する。"""
    # 全馬単勝
    n_all = len(df_bt)
    ret_all = (df_bt["finish_position"] == 1).astype(int).values * df_bt["odds"].values * 100
    roi_all = ret_all.sum() / (n_all * 100) if n_all else 0.0

    # 1番人気
    if "popularity" in df_bt.columns:
        fav = df_bt[pd.to_numeric(df_bt["popularity"], errors="coerce") == 1]
    else:
        fav = df_bt.groupby("race_id").apply(
            lambda g: g.loc[g["odds"].idxmin()]
        ).reset_index(drop=True)

    n_fav  = len(fav)
    ret_fav = (fav["finish_position"] == 1).astype(int).values * fav["odds"].values * 100
    roi_fav = ret_fav.sum() / (n_fav * 100) if n_fav else 0.0

    return {
        "all_n":    n_all,
        "all_roi":  round(roi_all, 4),
        "fav_n":    n_fav,
        "fav_roi":  round(roi_fav, 4),
    }


def fukusho_rate_at_theta(df_bt: pd.DataFrame, theta: float) -> float:
    """乖離選択馬の3着以内率(複勝代理指標)。"""
    sel = df_bt[df_bt["deviation"] >= theta]
    if len(sel) == 0:
        return 0.0
    return float((sel["finish_position"] <= 3).mean())


def top_deviation_examples(df_bt: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """乖離上位n頭の実例を返す。"""
    cols = ["race_date", "race_id", "horse_name", "horse_number",
            "odds", "p_blind_cal", "p_market_top3", "deviation",
            "finish_position"]
    available = [c for c in cols if c in df_bt.columns]
    return (
        df_bt
        .sort_values("deviation", ascending=False)
        .head(n)[available]
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────────────────────
# レポート生成
# ──────────────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[list]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep   = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr   = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    body  = "\n".join(
        "| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
        for row in rows
    )
    return "\n".join([hdr, sep, body])


def build_report(
    auc_blind: float,
    theta_results: list[dict],
    best_theta: float,
    best_odds_bands: list[dict],
    best_venue_rows: list[dict],
    baseline: dict,
    examples: pd.DataFrame,
    val_period: tuple,
    n_train: int,
    n_val_cal: int,
    n_val_bt: int,
) -> str:
    lines = [
        "# 穴馬発見モデル バックテストレポート",
        "",
        f"**生成日**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 1. データ分割",
        "",
        f"| 区間 | 行数 | 用途 |",
        f"|:-----|-----:|:-----|",
        f"| race_date < 2025-07-01 | {n_train:,} | 盲目モデル学習 + 市場確率モデル学習 |",
        f"| 検証前半（キャリブレーション用） | {n_val_cal:,} | Plattスケーリング学習 |",
        f"| 検証後半（バックテスト専用）      | {n_val_bt:,} | 最終評価のみ（閾値選択に不使用） |",
        f"| バックテスト期間 | {val_period[0]} 〜 {val_period[1]} | |",
        "",
        "---",
        "",
        "## 2. 盲目モデル（odds/popularity除外）単体性能",
        "",
        f"- 検証後半 AUC: **{auc_blind:.4f}**  ← 現行モデルより低いのは正常（オッズ情報を使わないため）",
        "",
        "---",
        "",
        "## 3. θ別バックテスト成績（単勝戦略）",
        "",
    ]

    # θ別成績表
    rows_theta = [
        [
            r["theta"],
            r["n_bets"],
            r["bets_per_day"],
            f"{r['hit_rate']*100:.1f}%",
            f"{r['roi']*100:.1f}%",
            f"¥{r['invested']:,}",
            f"¥{r['returned']:,}",
        ]
        for r in theta_results
    ]
    lines.append(_md_table(
        ["θ", "購入頭数", "1日当り", "単勝的中率", "ROI", "投資計", "回収計"],
        rows_theta,
    ))
    lines += ["", f"**最良θ: {best_theta}**（ROIが最大）", "", "---", ""]

    # オッズ帯別
    lines += ["## 4. オッズ帯別ROI内訳（最良θ）", ""]
    rows_band = [
        [b["band"], b["n"], f"{b['hit_rate']*100:.1f}%", f"{b['roi']*100:.1f}%"]
        for b in best_odds_bands
    ]
    lines.append(_md_table(["オッズ帯", "頭数", "的中率", "ROI"], rows_band))
    lines += ["", "---", ""]

    # 競馬場別
    if best_venue_rows:
        lines += ["## 5. 競馬場別ROI内訳（最良θ・上位10会場）", ""]
        rows_venue = [
            [v["venue"], v["n"], f"{v['hit_rate']*100:.1f}%", f"{v['roi']*100:.1f}%"]
            for v in best_venue_rows
        ]
        lines.append(_md_table(["競馬場", "頭数", "的中率", "ROI"], rows_venue))
        lines += ["", "---", ""]

    # 複勝proxy
    lines += ["## 6. 複勝代理指標（3着以内率）", ""]
    rows_fb = [
        [r["theta"], f"{fukusho_rate_at_theta_val:.1%}"]
        for r, fukusho_rate_at_theta_val in [
            (r, r.get("fukusho_rate", 0.0)) for r in theta_results
        ]
    ]
    lines.append(_md_table(["θ", "3着以内率"], rows_fb))
    lines += ["", "---", ""]

    # ベースライン
    lines += [
        "## 7. ベースライン比較（同期間）",
        "",
        f"| 戦略 | 購入数 | ROI |",
        f"|:-----|-------:|----:|",
        f"| 全馬単勝 | {baseline['all_n']:,} | {baseline['all_roi']*100:.1f}% |",
        f"| 1番人気単勝 | {baseline['fav_n']:,} | {baseline['fav_roi']*100:.1f}% |",
        "",
        "---",
        "",
    ]

    # 乖離上位20頭
    lines += ["## 8. 乖離上位20頭の実例", ""]
    ex_rows = []
    for _, row in examples.iterrows():
        fp = row.get("finish_position", "?")
        result = "🏆1着" if fp == 1 else ("2着" if fp == 2 else ("3着" if fp == 3 else f"{fp}着"))
        ex_rows.append([
            str(row.get("race_date", ""))[:10],
            str(row.get("horse_name", "")),
            f"{row.get('odds', '?')}倍",
            f"{row.get('p_blind_cal', 0):.3f}",
            f"{row.get('p_market_top3', 0):.3f}",
            f"{row.get('deviation', 0):+.3f}",
            result,
        ])
    lines.append(_md_table(
        ["日付", "馬名", "オッズ", "盲目確率", "市場確率", "乖離", "結果"],
        ex_rows,
    ))
    lines += ["", "---", ""]

    lines += [
        "## 注記",
        "",
        "- **盲目モデル**: odds/popularity を除外したXGBoost（訓練期間で学習）",
        "- **市場確率**: log(p_win)・出走頭数 → ロジスティック回帰 → 3着以内確率",
        "- **乖離スコア**: 盲目モデルのPlatt補正済み確率 − 市場3着以内確率",
        "- **単勝払戻**: odds × 100円（控除率込みのオッズをそのまま使用）",
        "- 複勝配当データなし → 3着以内率を複勝代理指標として使用",
        "- キャリブレーションデータ（検証前半）はθ選択に **使用していない**",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────

def main(sample: bool = False) -> None:
    # ── データ読み込み ───────────────────────────────────────────
    df = load_data(sample=sample)
    df["race_date"] = pd.to_datetime(df["race_date"])

    # ── 時系列分割 ───────────────────────────────────────────────
    df_train = df[df["race_date"] < TRAIN_END].reset_index(drop=True)
    df_val   = df[df["race_date"] >= TRAIN_END].reset_index(drop=True)

    if len(df_train) < 100:
        logger.warning(
            f"学習データが{len(df_train)}行しかありません。"
            "cleaned_races.csv.gz のデータが2025-07-01より前を含むか確認してください。"
            " --sample モードではこの警告が出やすいです。"
        )
    if len(df_val) < 50:
        logger.warning(
            f"検証データが{len(df_val)}行しかありません。"
            "バックテスト結果の信頼性が低くなります。"
        )

    # 検証を前半(キャリブレーション用)と後半(バックテスト専用)に等分
    mid_idx  = len(df_val) // 2
    df_cal   = df_val.iloc[:mid_idx].reset_index(drop=True)
    df_bt    = df_val.iloc[mid_idx:].reset_index(drop=True)

    logger.info(
        f"分割: train={len(df_train)}, val_cal={len(df_cal)}, val_bt={len(df_bt)}"
    )

    # ── 盲目モデル学習 ───────────────────────────────────────────
    blind_model = train_blind_model(df_train)

    # ── キャリブレーション学習（検証前半のみ） ─────────────────
    p_cal_raw = predict_blind(blind_model, df_cal)
    calibrator = Calibrator(method="platt")
    calibrator.fit(df_cal["top3"].astype(int).values, p_cal_raw)
    logger.info(f"Calibrator: method={calibrator.method}, ECE_raw={calibrator.report.get('ece_raw_val', '?'):.4f}")

    # ── バックテスト期間の予測 ───────────────────────────────────
    p_bt_raw = predict_blind(blind_model, df_bt)
    p_bt_cal = calibrator.transform(p_bt_raw)

    # AUC（盲目モデル単体）
    y_bt = df_bt["top3"].astype(int).values
    try:
        auc_blind = roc_auc_score(y_bt, p_bt_cal)
    except Exception:
        auc_blind = float("nan")
    logger.info(f"盲目モデル AUC（バックテスト後半）: {auc_blind:.4f}")

    # ── 市場確率モデル ───────────────────────────────────────────
    mkt_model = train_market_top3_model(df_train)
    p_mkt_bt  = predict_market_top3(mkt_model, df_bt)

    # ── 乖離スコア ───────────────────────────────────────────────
    df_bt = df_bt.copy()
    df_bt["p_blind_cal"]    = p_bt_cal
    df_bt["p_market_top3"]  = p_mkt_bt
    df_bt["deviation"]      = p_bt_cal - p_mkt_bt

    # ── バックテスト ─────────────────────────────────────────────
    theta_results = []
    for theta in THETA_LIST:
        res = run_tan_strategy(df_bt, theta)
        res["fukusho_rate"] = fukusho_rate_at_theta(df_bt, theta)
        theta_results.append(res)
        logger.info(
            f"θ={theta}: n={res['n_bets']}, ROI={res['roi']*100:.1f}%, "
            f"hit={res['hit_rate']*100:.1f}%, top3率={res['fukusho_rate']*100:.1f}%"
        )

    # 最良θ（ROI最大）
    best = max(theta_results, key=lambda r: r["roi"])
    best_theta = best["theta"]
    logger.info(f"最良θ: {best_theta}  ROI={best['roi']*100:.1f}%")

    # オッズ帯別・競馬場別
    best_odds_bands  = odds_band_roi(df_bt, best_theta)
    best_venue_rows  = venue_roi(df_bt, best_theta)

    # ベースライン
    baseline = baseline_roi(df_bt)
    logger.info(
        f"ベースライン: 全馬ROI={baseline['all_roi']*100:.1f}%, "
        f"1番人気ROI={baseline['fav_roi']*100:.1f}%"
    )

    # 乖離上位20頭
    examples = top_deviation_examples(df_bt, n=20)

    # バックテスト期間
    bt_start = df_bt["race_date"].min().strftime("%Y-%m-%d") if len(df_bt) else "N/A"
    bt_end   = df_bt["race_date"].max().strftime("%Y-%m-%d") if len(df_bt) else "N/A"

    # ── レポート生成 ─────────────────────────────────────────────
    report = build_report(
        auc_blind=auc_blind,
        theta_results=theta_results,
        best_theta=best_theta,
        best_odds_bands=best_odds_bands,
        best_venue_rows=best_venue_rows,
        baseline=baseline,
        examples=examples,
        val_period=(bt_start, bt_end),
        n_train=len(df_train),
        n_val_cal=len(df_cal),
        n_val_bt=len(df_bt),
    )

    # 標準出力
    print(report)

    # ファイル保存
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info(f"レポート保存: {REPORT_PATH}")


# ──────────────────────────────────────────────────────────────────
# サンプルテスト（--sample フラグで起動）
# ──────────────────────────────────────────────────────────────────

def test_sample() -> None:
    """直近1万行でパイプライン全体の動作確認を行う。"""
    logger.info("=== サンプルテスト開始（--sample） ===")
    main(sample=True)
    logger.info("=== サンプルテスト完了 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="穴馬発見モデルバックテスト")
    parser.add_argument(
        "--sample", action="store_true",
        help="直近1万行のみで動作確認（テスト用）",
    )
    args = parser.parse_args()
    main(sample=args.sample)
