"""
_best_encoding のユニットテスト。

euc-jp / utf-8 / cp932 それぞれでエンコードした日本語テキストに対して
正しいエンコーディングを採用し、置換文字(U+FFFD)が含まれないことを検証する。
"""

import re
import sys
from pathlib import Path

import pytest


def _best_encoding(content: bytes, hint: str, ct_charset: str) -> tuple[str, str]:
    """netkeiba_scraper._best_encoding の直接コピー（依存なしでテスト可能にする）。"""
    REPLACEMENT = "�"
    candidates: list[str] = []
    if ct_charset and not ct_charset.startswith("iso-8859"):
        candidates.append(ct_charset)
    if hint and hint.lower() not in [c.lower() for c in candidates]:
        candidates.append(hint)
    for enc in ("utf-8", "euc-jp", "cp932"):
        if enc not in [c.lower() for c in candidates]:
            candidates.append(enc)

    best_text = content.decode("utf-8", errors="replace")
    best_enc = "utf-8"
    best_count = best_text.count(REPLACEMENT)

    for enc in candidates:
        try:
            text = content.decode(enc, errors="replace")
        except (UnicodeDecodeError, LookupError):
            continue
        count = text.count(REPLACEMENT)
        if count < best_count:
            best_text = text
            best_enc = enc
            best_count = count
        if count == 0:
            return text, enc

    return best_text, best_enc

SAMPLE = "ダイワスカーレット　牝3　武豊　芝2000m　良"
REPLACEMENT = "�"


class TestBestEncoding:
    def test_utf8_content_no_hint(self):
        content = SAMPLE.encode("utf-8")
        text, enc = _best_encoding(content, hint="utf-8", ct_charset="")
        assert REPLACEMENT not in text
        assert SAMPLE in text

    def test_eucjp_content_with_eucjp_hint(self):
        content = SAMPLE.encode("euc-jp")
        text, enc = _best_encoding(content, hint="euc-jp", ct_charset="")
        assert REPLACEMENT not in text
        assert SAMPLE in text

    def test_cp932_content_with_cp932_hint(self):
        content = SAMPLE.encode("cp932")
        text, enc = _best_encoding(content, hint="cp932", ct_charset="")
        assert REPLACEMENT not in text
        assert SAMPLE in text

    def test_eucjp_content_utf8_hint_falls_back(self):
        """euc-jpコンテンツにutf-8ヒントを渡しても置換文字最少のものを採用する。"""
        content = SAMPLE.encode("euc-jp")
        text, enc = _best_encoding(content, hint="utf-8", ct_charset="")
        assert REPLACEMENT not in text
        assert SAMPLE in text

    def test_utf8_content_eucjp_hint_falls_back(self):
        """utf-8コンテンツにeuc-jpヒントを渡しても置換文字最少のものを採用する。"""
        content = SAMPLE.encode("utf-8")
        text, enc = _best_encoding(content, hint="euc-jp", ct_charset="")
        assert REPLACEMENT not in text
        assert SAMPLE in text

    def test_ct_charset_takes_priority(self):
        """Content-Typeのcharsetが正しければそれを最優先で採用する。"""
        content = SAMPLE.encode("utf-8")
        text, enc = _best_encoding(content, hint="euc-jp", ct_charset="utf-8")
        assert REPLACEMENT not in text
        assert enc.lower() == "utf-8"

    def test_iso8859_ct_charset_ignored(self):
        """iso-8859-* はノイズ検出として無視する。"""
        content = SAMPLE.encode("utf-8")
        text, enc = _best_encoding(content, hint="utf-8", ct_charset="iso-8859-1")
        assert REPLACEMENT not in text
        assert SAMPLE in text

    def test_eucjp_returns_eucjp_or_compatible(self):
        content = SAMPLE.encode("euc-jp")
        text, enc = _best_encoding(content, hint="euc-jp", ct_charset="")
        assert enc.lower() in ("euc-jp", "euc_jp", "euc_jis_2004", "cp932")

    def test_cp932_content_no_hint_auto_detect(self):
        """cp932固有文字（拡張）を含むテキストでも置換文字なしで復元できる。"""
        text_cp932 = "髙橋　㈱サンプル商事"  # cp932固有文字
        content = text_cp932.encode("cp932")
        text, enc = _best_encoding(content, hint="utf-8", ct_charset="")
        assert REPLACEMENT not in text

    def test_already_utf8_with_ct_charset(self):
        """nar.netkeiba.comがUTF-8に変わった後のシナリオ: Content-Typeにutf-8が来る。"""
        horse_name = "レッドファルクス"
        content = horse_name.encode("utf-8")
        text, enc = _best_encoding(content, hint="euc-jp", ct_charset="utf-8")
        assert horse_name in text
        assert REPLACEMENT not in text
