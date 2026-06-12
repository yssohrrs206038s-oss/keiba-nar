"""
モデルファイル (xgb_model.pkl) をダウンロードする。

優先順位:
  1. GitHub Releases の model-latest タグ（GITHUB_REPOSITORY 環境変数から URL を組み立て）
  2. Google Drive (gdown) へフォールバック（移行期の安全弁）

環境変数:
    GITHUB_REPOSITORY   : "owner/repo" 形式。GitHub Actions では自動設定される。
    NAR_MODEL_GDRIVE_ID : フォールバック用 Google Drive ファイル ID（省略可）

使い方:
    python keiba_predictor/scripts/download_model.py
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR  = Path(__file__).parent.parent / "model"
MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

RELEASE_TAG  = "model-latest"
ASSET_NAME   = "xgb_model.pkl"


def _github_release_url(repo: str) -> str:
    """GitHub Releases の直接ダウンロード URL を組み立てる。"""
    return f"https://github.com/{repo}/releases/download/{RELEASE_TAG}/{ASSET_NAME}"


def _download_from_github(repo: str) -> bool:
    """
    GitHub Releases から MODEL_PATH へダウンロードする。
    成功なら True、失敗なら False を返す。
    """
    url = _github_release_url(repo)
    print(f"[GitHub Releases] ダウンロード中: {url}", flush=True)
    tmp = MODEL_PATH.with_suffix(".tmp")
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(MODEL_PATH)
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"[GitHub Releases] 完了: {MODEL_PATH} ({size_mb:.1f}MB)", flush=True)
        return True
    except Exception as e:
        print(f"[GitHub Releases] 失敗: {e}", flush=True)
        if tmp.exists():
            tmp.unlink()
        return False


def _download_from_gdrive(gdrive_id: str) -> bool:
    """
    Google Drive から MODEL_PATH へダウンロードする。
    成功なら True、失敗なら False を返す。
    """
    try:
        import gdown
    except ImportError:
        print("[GDrive] gdown 未インストールのためスキップ", flush=True)
        return False

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[GDrive] フォールバック中: {url}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(url, str(MODEL_PATH), quiet=False)
    except Exception as e:
        print(f"[GDrive] 失敗: {e}", flush=True)
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        return False

    if not MODEL_PATH.exists():
        print("[GDrive] ダウンロード後にファイルが見つかりません", flush=True)
        return False

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"[GDrive] 完了: {MODEL_PATH} ({size_mb:.1f}MB)", flush=True)
    return True


def download_model() -> None:
    # 既にファイルが存在する場合はスキップ
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"モデルファイル既存: {MODEL_PATH} ({size_mb:.1f}MB) → スキップ", flush=True)
        return

    repo = os.environ.get("GITHUB_REPOSITORY", "")

    # 1. GitHub Releases を試みる
    if repo:
        if _download_from_github(repo):
            return
        print("[GitHub Releases] 失敗 → GDrive フォールバックを試みます", flush=True)
    else:
        print(
            "GITHUB_REPOSITORY 未設定 → GDrive フォールバックを試みます",
            flush=True,
        )

    # 2. GDrive フォールバック
    gdrive_id = os.environ.get("NAR_MODEL_GDRIVE_ID", "")
    if gdrive_id:
        if _download_from_gdrive(gdrive_id):
            return
    else:
        print("[GDrive] NAR_MODEL_GDRIVE_ID 未設定のためスキップ", flush=True)

    print(
        "ERROR: GitHub Releases / GDrive の両方からダウンロードに失敗しました。",
        flush=True,
    )
    sys.exit(1)


if __name__ == "__main__":
    download_model()
