"""
新特徴量（クラス昇降・Elo・騎手乗り替わり）の効果測定スクリプト。

出力: keiba_predictor/data/feature_eval_report.md
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR   = ROOT / "keiba_predictor" / "data"
MODEL_DIR  = ROOT / "keiba_predictor" / "model"
REPORT_PATH = DATA_DIR / "feature_eval_report.md"

TRAIN_END  = pd.Timestamp("2024-07-01")
VAL_START  = pd.Timestamp("2024-07-01")
VAL_END    = pd.Timestamp("2026-01-01")

OLD_FEATURE_COLS_49 = [
    "distance", "course_type_enc", "track_condition_enc", "weather_enc",
    "frame_number", "horse_number", "weight_carried", "odds", "popularity",
    "sex_enc", "age", "horse_weight", "horse_weight_diff", "last_3f",
    "avg_time_3", "avg_time_5", "avg_time_3_any", "avg_time_5_any",
    "jockey_fukusho_rate", "trainer_fukusho_rate", "dist_diff_prev",
    "days_since_last_race", "prev_finish_pos", "prev_odds",
    "prev2_finish_pos", "prev2_odds", "prev3_finish_pos",
    "prev2_last_3f", "prev3_last_3f", "finish_pos_trend",
    "horse_course_fukusho_rate", "horse_dist_fukusho_rate",
    "jockey_horse_fukusho_rate", "horse_track_fukusho_rate",
    "jockey_district_fukusho_rate", "jockey_dist_fukusho_rate",
    "weeks_since_last_race", "is_fresh", "is_continuous",
    "jockey_trainer_fukusho_rate", "weight_carried_diff", "is_weight_increase",
    "same_day_rank", "prob_vs_avg",
    "sire_win_rate", "bms_win_rate", "sire_course_win_rate",
    "sire_dist_win_rate", "bms_course_win_rate",
]

NEW_FEATURE_COLS = [
    "race_class_level", "prev_class_level", "class_diff",
    "is_class_up", "is_class_down",
    "horse_elo", "elo_minus_field_avg", "prev_race_opp_elo",
    "is_jockey_change",
]


def _load_data(sample: bool = False) -> pd.DataFrame:
    """featured_races.csv があればそこから、なければ cleaned_races.csv.gz から全特徴量を構築する。"""
    featured_path = DATA_DIR / "featured_races.csv"
    cleaned_gz    = DATA_DIR / "cleaned_races.csv.gz"
    cleaned_csv   = DATA_DIR / "cleaned_races.csv"

    if featured_path.exists():
        print(f"featured_races.csv 読み込み中: {featured_path}")
        df = pd.read_csv(featured_path, encoding="utf-8-sig", low_memory=False)
        df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        # 新特徴量が欠けていれば追加
        missing = [c for c in NEW_FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"新特徴量を追加中: {missing}")
            from keiba_predictor.features.feature_engineering import (
                add_race_class_features, add_elo_features, add_jockey_change_feature
            )
            df = add_race_class_features(df)
            df = add_elo_features(df)
            df = add_jockey_change_feature(df)
        return df

    # featured_races.csv なし → cleaned_races.csv.gz から構築
    if cleaned_csv.exists():
        src = cleaned_csv
    elif cleaned_gz.exists():
        src = cleaned_gz
    else:
        print("データファイルが見つかりません。")
        sys.exit(1)

    print(f"クリーニング済みデータ読み込み中: {src}")
    df = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    if sample:
        # 訓練・検証をカバーするため 2024-01-01 以降のみ (≈最終 10 万行程度)
        df_recent = df[df["race_date"] >= "2023-01-01"]
        if len(df_recent) > 0:
            df = df_recent
        print(f"サンプルモード: {len(df):,} 行に絞り込み")

    print("全特徴量を構築中（初回は時間がかかります）...")
    from keiba_predictor.features.feature_engineering import build_features
    df = build_features(df)
    return df


_LEAKY_COLS = {"odds", "popularity", "last_3f", "same_day_rank", "prob_vs_avg"}


def _train_xgb(X_train, y_train, X_val, y_val):
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    return model, auc


def _blind_roi(df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: list[str]) -> float:
    """オッズなし特徴量でブラインドモデルを学習しバックテスト ROI を返す（単勝1点買い）。"""
    blind_cols = [c for c in feature_cols if c not in _LEAKY_COLS]
    X_tr = df_train[blind_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X_vl = df_val[blind_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_tr = df_train["top3"]
    y_vl = df_val["top3"]
    blind_model, _ = _train_xgb(X_tr, y_tr, X_vl, y_vl)
    probs = blind_model.predict_proba(X_vl)[:, 1]

    df_eval = df_val[["race_id", "finish_position", "odds"]].copy()
    df_eval["prob"] = probs
    pnl = []
    for _, race in df_eval.groupby("race_id", sort=False):
        best_idx = race["prob"].idxmax()
        win_odds = pd.to_numeric(race.loc[best_idx, "odds"], errors="coerce")
        finish   = pd.to_numeric(race.loc[best_idx, "finish_position"], errors="coerce")
        if pd.isna(win_odds) or pd.isna(finish):
            continue
        pnl.append(float(win_odds) if finish == 1 else 0.0)
    if not pnl:
        return float("nan")
    return float(np.mean(pnl))


def main(sample: bool = False):
    df = _load_data(sample=sample)

    # top3 ラベル
    df["top3"] = pd.to_numeric(df.get("top3", df.get("finish_position")), errors="coerce").le(3).astype(int)
    if "finish_position" in df.columns:
        df["top3"] = pd.to_numeric(df["finish_position"], errors="coerce").le(3).astype(int)

    train_df = df[df["race_date"] < TRAIN_END].copy()
    val_df   = df[(df["race_date"] >= VAL_START) & (df["race_date"] < VAL_END)].copy()
    print(f"訓練: {len(train_df):,} 行 / 検証: {len(val_df):,} 行")

    ALL_COLS = list(dict.fromkeys(OLD_FEATURE_COLS_49 + NEW_FEATURE_COLS))
    old_cols = [c for c in OLD_FEATURE_COLS_49
                if c not in ("same_day_rank", "prob_vs_avg") and c in df.columns]
    new_cols = [c for c in ALL_COLS
                if c not in ("same_day_rank", "prob_vs_avg") and c in df.columns]

    def _prep(d, cols):
        return d[cols].apply(pd.to_numeric, errors="coerce").fillna(0), d["top3"]

    print("\n旧特徴量(49)でモデル学習中...")
    X_tr_old, y_tr = _prep(train_df, old_cols)
    X_val_old, y_val = _prep(val_df, old_cols)
    model_old, auc_old = _train_xgb(X_tr_old, y_tr, X_val_old, y_val)
    print("旧ブラインドROI計算中...")
    roi_old = _blind_roi(train_df, val_df, old_cols)
    print(f"旧モデル AUC={auc_old:.4f}  ROI={roi_old:.3f}")

    print("\n新特徴量(58)でモデル学習中...")
    X_tr_new, _ = _prep(train_df, new_cols)
    X_val_new, _ = _prep(val_df, new_cols)
    model_new, auc_new = _train_xgb(X_tr_new, y_tr, X_val_new, y_val)
    print("新ブラインドROI計算中...")
    roi_new = _blind_roi(train_df, val_df, new_cols)
    print(f"新モデル AUC={auc_new:.4f}  ROI={roi_new:.3f}")

    # Feature importance top 20 (新特徴量に絞る)
    import xgboost as xgb
    importance = model_new.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame(list(importance.items()), columns=["feature", "gain"])
    imp_df = imp_df.sort_values("gain", ascending=False).head(20).reset_index(drop=True)

    # 新特徴量のみの importance
    new_imp = imp_df[imp_df["feature"].isin(NEW_FEATURE_COLS)].copy()

    lines = [
        "# 特徴量追加効果測定レポート",
        "",
        f"生成日: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"訓練期間: ～ {TRAIN_END.date()} / 検証期間: {VAL_START.date()} ～ {VAL_END.date()}",
        f"訓練行数: {len(train_df):,} / 検証行数: {len(val_df):,}",
        "",
        "## AUC 比較",
        "",
        f"| モデル | 特徴量数 | Val AUC |",
        f"|--------|----------|---------|",
        f"| 旧     | {len(old_cols)} | {auc_old:.4f} |",
        f"| 新     | {len(new_cols)} | {auc_new:.4f} |",
        f"| 差     |          | {auc_new - auc_old:+.4f} |",
        "",
        "## ブラインドモデル ROI 比較（単勝1点買い、Val期間）",
        "",
        f"| モデル | ROI  |",
        f"|--------|------|",
        f"| 旧     | {roi_old:.3f} |",
        f"| 新     | {roi_new:.3f} |",
        f"| 差     | {roi_new - roi_old:+.3f} |",
        "",
        "## XGBoost 特徴量重要度 Top 20（新モデル）",
        "",
        "| Rank | Feature | Gain |",
        "|------|---------|------|",
    ]
    for i, row in imp_df.iterrows():
        marker = " ★" if row["feature"] in NEW_FEATURE_COLS else ""
        lines.append(f"| {i+1} | {row['feature']}{marker} | {row['gain']:.1f} |")

    lines += [
        "",
        "（★ = 今回追加した新特徴量）",
        "",
        "## 新特徴量のみの重要度",
        "",
        "| Feature | Gain |",
        "|---------|------|",
    ]
    for _, row in new_imp.iterrows():
        lines.append(f"| {row['feature']} | {row['gain']:.1f} |")
    if new_imp.empty:
        lines.append("| （Top 20 未ランクイン） | — |")

    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nレポート保存: {REPORT_PATH}")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="直近20,000行のみ使用")
    args = parser.parse_args()
    main(sample=args.sample)
