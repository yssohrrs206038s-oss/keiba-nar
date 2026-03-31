"""
Google Drive から NAR モデルファイルをダウンロードする。

環境変数:
    NAR_MODEL_GDRIVE_ID: Google Drive のファイル共有ID

使い方:
    python keiba_predictor/scripts/download_model.py
"""

import os
import sys
from pathlib import Path


MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"


def download_model() -> None:
    gdrive_id = os.environ.get("NAR_MODEL_GDRIVE_ID", "")
    if not gdrive_id:
        print("ERROR: NAR_MODEL_GDRIVE_ID が設定されていません。", flush=True)
        print("GitHub Secrets に Google Drive のファイルIDを登録してください。", flush=True)
        sys.exit(1)

    # 既にファイルが存在する場合はスキップ
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"モデルファイル既存: {MODEL_PATH} ({size_mb:.1f}MB) → スキップ", flush=True)
        return

    try:
        import gdown
    except ImportError:
        print("ERROR: gdown がインストールされていません: pip install gdown", flush=True)
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"モデルダウンロード中: {url}", flush=True)
    print(f"保存先: {MODEL_PATH}", flush=True)

    try:
        gdown.download(url, str(MODEL_PATH), quiet=False)
    except Exception as e:
        print(f"ERROR: ダウンロード失敗: {e}", flush=True)
        # 不完全なファイルがあれば削除
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        sys.exit(1)

    if not MODEL_PATH.exists():
        print("ERROR: ダウンロードしたファイルが見つかりません。", flush=True)
        print("Google Drive の共有設定を確認してください（リンクを知っている全員に公開）。", flush=True)
        sys.exit(1)

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"ダウンロード完了: {MODEL_PATH} ({size_mb:.1f}MB)", flush=True)


if __name__ == "__main__":
    download_model()
