"""
download_model.py の URL 組み立てロジックのユニットテスト。
ネットワーク不要。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from keiba_predictor.scripts.download_model import _github_release_url, RELEASE_TAG, ASSET_NAME


class TestGithubReleaseUrl:
    def test_standard_repo(self):
        url = _github_release_url("owner/repo")
        assert url == "https://github.com/owner/repo/releases/download/model-latest/xgb_model.pkl"

    def test_org_repo(self):
        url = _github_release_url("my-org/keiba-nar")
        assert url.startswith("https://github.com/my-org/keiba-nar/releases/download/")
        assert RELEASE_TAG in url
        assert ASSET_NAME in url

    def test_contains_tag(self):
        url = _github_release_url("a/b")
        assert f"/releases/download/{RELEASE_TAG}/" in url

    def test_contains_asset(self):
        url = _github_release_url("a/b")
        assert url.endswith(f"/{ASSET_NAME}")

    def test_no_double_slash(self):
        url = _github_release_url("owner/repo")
        # github.com の後にスラッシュが重複しないこと
        assert "///" not in url

    def test_constants_unchanged(self):
        """RELEASE_TAG と ASSET_NAME が意図した値であること。"""
        assert RELEASE_TAG == "model-latest"
        assert ASSET_NAME == "xgb_model.pkl"
