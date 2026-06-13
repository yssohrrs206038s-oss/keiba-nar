"""
モデルファイルを GitHub Releases からダウンロードする。

対象ファイル:
  xgb_model.pkl          (必須 — 失敗時は GDrive フォールバック)
  calibrator.pkl         (任意)
  xgb_model_sprint.pkl   (任意)
  xgb_model_mile.pkl     (任意)
  xgb_model_middle.pkl   (任意)
  xgb_model_long.pkl     (任意)

環境変数:
    GITHUB_REPOSITORY   : "owner/repo" 形式。GitHub Actions では自動設定される。
    NAR_MODEL_GDRIVE_ID : 主モデルのフォールバック用 Google Drive ファイル ID（省略可）

使い方:
    python keiba_predictor/scripts/download_model.py
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model"

RELEASE_TAG = "model-latest"

# 必須モデル（失敗時 GDrive フォールバック）
MAIN_ASSET = "xgb_model.pkl"
# 任意モデル（欠損は警告のみ）
OPTIONAL_ASSETS = [
    "calibrator.pkl",
    "xgb_model_sprint.pkl",
    "xgb_model_mile.pkl",
    "xgb_model_middle.pkl",
    "xgb_model_long.pkl",
]


def _release_url(repo: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{RELEASE_TAG}/{asset}"


def _download_asset(repo: str, asset: str, *, required: bool = True) -> bool:
    dest = MODEL_DIR / asset
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"既存スキップ: {asset} ({size_mb:.1f}MB)", flush=True)
        return True

    url = _release_url(repo, asset)
    tmp = dest.with_suffix(".tmp")
    print(f"[GitHub Releases] ダウンロード中: {url}", flush=True)
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[GitHub Releases] 完了: {asset} ({size_mb:.1f}MB)", flush=True)
        return True
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        label = "::warning::" if not required else ""
        print(f"{label}[GitHub Releases] {asset} 失敗: {e}", flush=True)
        return False


def _download_from_gdrive(gdrive_id: str) -> bool:
    dest = MODEL_DIR / MAIN_ASSET
    try:
        import gdown
    except ImportError:
        print("[GDrive] gdown 未インストールのためスキップ", flush=True)
        return False

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[GDrive] フォールバック中: {url}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(url, str(dest), quiet=False)
    except Exception as e:
        print(f"[GDrive] 失敗: {e}", flush=True)
        if dest.exists():
            dest.unlink()
        return False

    if not dest.exists():
        print("[GDrive] ダウンロード後にファイルが見つかりません", flush=True)
        return False

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"[GDrive] 完了: {MAIN_ASSET} ({size_mb:.1f}MB)", flush=True)
    return True


def download_model() -> None:
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    # ── 必須モデル ─────────────────────────────────────────
    main_ok = False
    if repo:
        main_ok = _download_asset(repo, MAIN_ASSET, required=True)
        if not main_ok:
            print("[GitHub Releases] 失敗 → GDrive フォールバックを試みます", flush=True)
    else:
        print("GITHUB_REPOSITORY 未設定 → GDrive フォールバックを試みます", flush=True)

    if not main_ok:
        gdrive_id = os.environ.get("NAR_MODEL_GDRIVE_ID", "")
        if gdrive_id:
            main_ok = _download_from_gdrive(gdrive_id)
        else:
            print("[GDrive] NAR_MODEL_GDRIVE_ID 未設定のためスキップ", flush=True)

    if not main_ok:
        print("ERROR: xgb_model.pkl のダウンロードに失敗しました。", flush=True)
        sys.exit(1)

    # ── 任意モデル（距離帯別・キャリブレータ） ────────────
    if repo:
        for asset in OPTIONAL_ASSETS:
            _download_asset(repo, asset, required=False)
    else:
        print("GITHUB_REPOSITORY 未設定 → 距離帯別モデルのダウンロードをスキップ", flush=True)


if __name__ == "__main__":
    download_model()
