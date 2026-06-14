# 実装指示書: 全頭予測ログ + 配当テーブル保存

対象リポジトリ: keiba-nar (main) / keiba-predictor (claude/horse-racing-predictor-TLZas)
両方に同じ仕組みを実装する。NARを先に実装し、動作確認後にJRAへ移植すること。

## 目的

現状のresults_history.csvは上位3頭の確率しか残していないため、(1) 全確率帯でのキャリブレーション学習、(2) 市場乖離モデル(自モデル確率−オッズ逆算の市場確率)の学習、(3) 任意の買い目戦略の後付けバックテスト、ができない。オッズと予測値は記録しなければ消えるデータなので、全頭分のログ基盤を作る。

## 成果物1: 全頭予測ログ (prediction_log.csv)

保存先: `keiba_predictor/data/prediction_log.csv`(追記式)

1行 = 1頭。カラム:

| カラム | 内容 |
|---|---|
| race_date | YYYY-MM-DD |
| race_id | レースID |
| horse_number | 馬番 |
| horse_name | 馬名 |
| prob_raw | モデル生出力 (prob_top3) |
| prob_cal | 補正後確率 (prob_top3_cal)。calibrator未適用時は空 |
| odds_predict | 予測実行時点のオッズ |
| popularity | 予測時点の人気 |
| mark | 印 (◎○▲△ または空) |
| finish_position | 着順(結果処理時に更新。予測時は空) |
| odds_final | 確定単勝オッズ(結果処理時に更新。取得可能な場合のみ) |

実装ポイント:
- 書き込みタイミングは2回。①予測ステップ(predict実行時)で finish_position/odds_final 以外を追記、②結果処理ステップ(record_result呼び出し付近)で該当race_idの行の finish_position / odds_final を更新
- 同一race_idの重複追記を防ぐこと(予測が複数回走るケースがある。race_id+horse_numberをキーに既存行があればスキップ or 上書き)
- 既存のpredict処理・買い目ロジック・通知には一切影響を与えないこと。ログ書き込みの失敗は warning ログを出して握りつぶす(本処理を止めない)
- 追記時はCSVロックを考慮しなくてよい(Actionsは直列実行)
- エンコーディングは utf-8 固定

## 成果物2: 配当テーブル全種の保存 (payouts_log)

保存先: `keiba_predictor/data/payouts/` ディレクトリに `{race_id}.json`(1レース1ファイル)

- 結果処理時に取得済みの payouts dict をそのままJSON保存する(再スクレイピング不要、すでにメモリにあるデータを書くだけ)
- ensure_ascii=False, indent=2
- 既存ファイルがあれば上書きしない(冪等)
- .gitignore には入れない(リポジトリにコミットして蓄積する)。1ファイル1KB未満×年間数千件程度で問題ない

## 実装箇所のヒント(調査済み)

- NAR predict: `keiba_predictor/model/predict.py` の `predict_race()` が prob_top3 / prob_top3_cal を持つDataFrameを返す。印・買い目は `_decide_bet_strategy()`。予測のエントリポイントとキャッシュ書き込みは main.py / discord_notify.py 側にある
- NAR 結果処理: `keiba_predictor/discord_notify.py` 内の結果通知ループで `scrape_race_result()` → `record_result()`(history.py) が呼ばれる。payouts dict はこの時点で手元にある
- JRAも同名ファイル・類似構造(完全同一ではないので個別確認)

## 制約(重要)

- 買い目・通知・既存CSVのスキーマと挙動を変えないこと。今回は「記録の追加」のみ
- prob_top3(生確率)とprob_top3_cal(補正後)を取り違えないこと。両方記録する
- 直近で入った変更と競合しないこと: calibration.py導入、多頭数シャドウ(shadow_type列)、エンコーディング自動フォールバック

## テスト要件

- 予測→ログ追記→同レース再予測(重複しないこと)→結果処理→finish_position更新、の一連をモックまたはダミーデータで通すこと
- 既存テスト・py_compileが全ファイル通ること
- ログ書き込み部で例外を強制発生させても本処理が完走することを確認

## 完了条件

NARで1開催日分が実運用でログされたのを確認後、JRAに移植。実装完了後の差分はCodexがレビューする前提で、コミットメッセージに変更理由と検証内容を書くこと。
