"""
学習済みモデルを Google Drive にアップロードする。

環境変数:
    GDRIVE_SERVICE_ACCOUNT_JSON: サービスアカウントの JSON 鍵（文字列）
    NAR_MODEL_GDRIVE_ID: Google Drive のファイル ID

使い方:
    python keiba_predictor/scripts/upload_model.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / "model" / "xgb_model.pkl"


def upload_model() -> bool:
    sa_json = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON", "")
    gdrive_id = os.environ.get("NAR_MODEL_GDRIVE_ID", "")

    if not sa_json:
        print("GDRIVE_SERVICE_ACCOUNT_JSON 未設定 → アップロードスキップ", flush=True)
        return False

    if not gdrive_id:
        print("NAR_MODEL_GDRIVE_ID 未設定 → アップロードスキップ", flush=True)
        return False

    if not MODEL_PATH.exists():
        print(f"ERROR: モデルファイルが見つかりません: {MODEL_PATH}", flush=True)
        return False

    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"アップロード対象: {MODEL_PATH} ({size_mb:.1f}MB)", flush=True)

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.oauth2 import service_account
    except ImportError as e:
        print(f"ERROR: Google API ライブラリ未インストール: {e}", flush=True)
        return False

    # サービスアカウントキーを一時ファイルに書き出し
    try:
        sa_data = json.loads(sa_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: GDRIVE_SERVICE_ACCOUNT_JSON の JSON パース失敗: {e}", flush=True)
        return False

    sa_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sa_data, f)
            sa_path = f.name

        creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        service = build("drive", "v3", credentials=creds)

        media = MediaFileUpload(str(MODEL_PATH), resumable=True)
        service.files().update(
            fileId=gdrive_id,
            media_body=media,
        ).execute()

        print(f"Google Drive アップロード完了: file_id={gdrive_id}", flush=True)
        return True

    except Exception as e:
        print(f"ERROR: Google Drive アップロード失敗: {e}", flush=True)
        return False

    finally:
        if sa_path and os.path.exists(sa_path):
            os.unlink(sa_path)


if __name__ == "__main__":
    ok = upload_model()
    # GitHub Actions の GITHUB_ENV にアップロード結果を保存
    env_file = os.environ.get("GITHUB_ENV")
    if env_file:
        with open(env_file, "a") as f:
            f.write(f"GDRIVE_UPLOAD={'成功' if ok else '失敗/スキップ'}\n")
    sys.exit(0)  # continue-on-error のため常に0で終了
