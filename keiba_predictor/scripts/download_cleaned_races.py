"""
Google Drive から cleaned_races.csv.gz をダウンロードし gunzip する。

環境変数:
    CLEANED_RACES_GDRIVE_ID: Google Drive のファイル共有ID

使い方:
    python keiba_predictor/scripts/download_cleaned_races.py
"""

import gzip
import os
import shutil
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
GZ_PATH = DATA_DIR / "cleaned_races.csv.gz"
CSV_PATH = DATA_DIR / "cleaned_races.csv"


def download_cleaned_races() -> None:
    gdrive_id = os.environ.get("CLEANED_RACES_GDRIVE_ID", "")
    if not gdrive_id:
        print("ERROR: CLEANED_RACES_GDRIVE_ID が設定されていません。", flush=True)
        print("GitHub Secrets に Google Drive のファイルIDを登録してください。", flush=True)
        sys.exit(1)

    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
        print(f"cleaned_races.csv 既存: {CSV_PATH} ({size_mb:.1f}MB) → スキップ", flush=True)
        return

    try:
        import gdown
    except ImportError:
        print("ERROR: gdown がインストールされていません: pip install gdown", flush=True)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"cleaned_races.csv.gz ダウンロード中: {url}", flush=True)
    print(f"保存先: {GZ_PATH}", flush=True)

    try:
        gdown.download(url, str(GZ_PATH), quiet=False)
    except Exception as e:
        print(f"ERROR: ダウンロード失敗: {e}", flush=True)
        if GZ_PATH.exists():
            GZ_PATH.unlink()
        sys.exit(1)

    if not GZ_PATH.exists():
        print("ERROR: ダウンロードしたファイルが見つかりません。", flush=True)
        print("Google Drive の共有設定を確認してください（リンクを知っている全員に公開）。", flush=True)
        sys.exit(1)

    print(f"gunzip 展開中: {GZ_PATH} → {CSV_PATH}", flush=True)
    try:
        with gzip.open(GZ_PATH, "rb") as f_in, open(CSV_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        print(f"ERROR: gunzip 展開失敗: {e}", flush=True)
        if CSV_PATH.exists():
            CSV_PATH.unlink()
        sys.exit(1)

    size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
    print(f"展開完了: {CSV_PATH} ({size_mb:.1f}MB)", flush=True)


if __name__ == "__main__":
    download_cleaned_races()
