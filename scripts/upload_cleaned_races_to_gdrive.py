"""
cleaned_races.csv.gz を Google Drive にアップロードするローカル実行スクリプト。

前提:
    pip install PyDrive2
    Google Cloud Console で OAuth クライアントID(デスクトップアプリ)を作成し、
    client_secrets.json をこのスクリプトと同じディレクトリに配置する。

使い方:
    python scripts/upload_cleaned_races_to_gdrive.py [CSV_GZ_PATH]

    引数省略時は ../keiba-predictor/keiba_predictor/data/cleaned_races.csv.gz を使用。

実行後、表示される Google Drive ファイルIDを GitHub Secrets の
CLEANED_RACES_GDRIVE_ID に登録してください。
共有設定は「リンクを知っている全員（閲覧者）」にしてください。
"""

import sys
from pathlib import Path


DEFAULT_GZ_PATH = Path(
    r"C:\Users\journ\keiba-predictor\keiba_predictor\data\cleaned_races.csv.gz"
)
REMOTE_NAME = "cleaned_races.csv.gz"


def main() -> None:
    gz_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_GZ_PATH
    if not gz_path.exists():
        print(f"ERROR: ファイルが見つかりません: {gz_path}", flush=True)
        sys.exit(1)

    size_mb = gz_path.stat().st_size / (1024 * 1024)
    print(f"アップロード対象: {gz_path} ({size_mb:.1f}MB)", flush=True)

    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        print("ERROR: PyDrive2 が未インストールです: pip install PyDrive2", flush=True)
        sys.exit(1)

    # client_secrets.json はこのスクリプトと同じディレクトリに配置
    script_dir = Path(__file__).parent
    client_secrets = script_dir / "client_secrets.json"
    if not client_secrets.exists():
        print(f"ERROR: client_secrets.json が見つかりません: {client_secrets}", flush=True)
        print("Google Cloud Console でデスクトップアプリ用 OAuth クライアントを作成して配置してください。", flush=True)
        sys.exit(1)

    gauth = GoogleAuth()
    gauth.settings["client_config_file"] = str(client_secrets)
    # 認証情報をローカルに保存（2回目以降は自動）
    creds_path = script_dir / "gdrive_credentials.json"
    gauth.LoadCredentialsFile(str(creds_path))
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(str(creds_path))

    drive = GoogleDrive(gauth)

    # 同名既存ファイルを検索 → あれば上書き、なければ新規
    query = f"title='{REMOTE_NAME}' and trashed=false"
    existing = drive.ListFile({"q": query}).GetList()

    if existing:
        gfile = existing[0]
        print(f"既存ファイルを上書き: id={gfile['id']}", flush=True)
    else:
        gfile = drive.CreateFile({"title": REMOTE_NAME})
        print("新規ファイルを作成", flush=True)

    gfile.SetContentFile(str(gz_path))
    gfile.Upload()

    # リンクを知っている全員に公開
    try:
        gfile.InsertPermission({
            "type": "anyone",
            "value": "anyone",
            "role": "reader",
        })
    except Exception as e:
        print(f"WARN: 公開設定失敗（手動で設定してください）: {e}", flush=True)

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"✅ アップロード完了", flush=True)
    print(f"   ファイルID: {gfile['id']}", flush=True)
    print(f"   GitHub Secrets に CLEANED_RACES_GDRIVE_ID として登録してください。", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
