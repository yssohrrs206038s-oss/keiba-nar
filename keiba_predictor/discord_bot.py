"""
Discord ボット（コマンド対応）

コマンド:
    !予想           本日の全特別レース予想
    !予想 笠松      指定開催場の本日予想
    !結果           本日の結果サマリー

環境変数:
    DISCORD_BOT_TOKEN : Discord Bot Token（Bot設定画面で取得）
"""

import json
import logging
import os
import re
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
PRED_CACHE = DATA_DIR / "predictions_cache.json"
HIST_PATH = DATA_DIR / "results_history.csv"

# 特別レース判定キーワード
_SPECIAL_KEYWORDS = ("特別", "記念", "杯", "賞", "ステークス")


def _load_cache() -> dict:
    if PRED_CACHE.exists():
        with open(PRED_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _today_entries(cache: dict, venue: str = "") -> list[tuple[str, dict]]:
    """本日のキャッシュエントリを返す。venue指定時はその開催場のみ。"""
    today_str = date.today().isoformat()
    entries = []
    for rid, entry in cache.items():
        if rid.startswith("_"):
            continue
        if entry.get("race_date") != today_str:
            continue
        if venue and entry.get("venue", "") != venue:
            continue
        entries.append((rid, entry))
    # 発走時刻順
    entries.sort(key=lambda x: x[1].get("start_time", "99:99"))
    return entries


def _format_prediction(entry: dict) -> str:
    """1レース分の予想テキストを生成。"""
    sep = "━" * 20
    race_name = entry.get("race_name", "")
    venue = entry.get("venue", "")
    start_time = entry.get("start_time", "")
    course_info = entry.get("course_info", "")

    meta_parts = []
    if course_info:
        meta_parts.append(course_info)
    if start_time:
        meta_parts.append(f"{start_time}発走")
    meta = " ".join(meta_parts)

    lines = [f"🏇 {race_name} {meta}"]

    for role, mark in [("honmei", "◎"), ("taikou", "○"), ("ana", "▲")]:
        p = entry.get(role, {})
        if not p or not p.get("horse_name"):
            continue
        num = p.get("horse_number", "?")
        name = p.get("horse_name", "")
        prob = p.get("prob", 0) * 100
        lines.append(f"{mark}{num}番 {name} {prob:.1f}%")

    # 買い目（ワイド ◎-○ 1点 1,000円のみ）
    top3 = entry.get("predicted_top3_nums", [])
    if len(top3) >= 2:
        lines.append(f"💰ワイド ◎{top3[0]}-○{top3[1]}（1点 1,000円）")

    return "\n".join(lines)


def _format_result_summary() -> str:
    """本日の結果サマリーを生成。"""
    if not HIST_PATH.exists():
        return "結果データがありません。"

    import pandas as pd
    df = pd.read_csv(HIST_PATH, encoding="utf-8-sig", dtype=str)
    today_str = date.today().isoformat()
    today_df = df[df["date"] == today_str]

    if today_df.empty:
        return f"本日（{today_str}）の結果はまだありません。"

    sep = "━" * 20
    lines = [f"🏆 本日のNAR結果 ({today_str})", sep]

    total = len(today_df)
    w_hits = (today_df.get("wide_hit", "False") == "True").sum()

    for _, r in today_df.iterrows():
        name = str(r.get("race_name", ""))[:20]
        w_icon = "✅" if r.get("wide_hit") == "True" else "❌"
        lines.append(f"{name} {w_icon}")

    lines.append(sep)
    lines.append(f"ワイド {w_hits}/{total} ({w_hits/total*100:.0f}%)")

    bet = pd.to_numeric(today_df["bet_total"], errors="coerce").sum()
    ret = pd.to_numeric(today_df["return_total"], errors="coerce").sum()
    roi = (ret / bet * 100) if bet > 0 else 0
    lines.append(f"回収率 {roi:.0f}%")

    return "\n".join(lines)


def run_bot():
    """Discord ボットを起動する。"""
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.error("DISCORD_BOT_TOKEN が設定されていません")
        print("DISCORD_BOT_TOKEN を環境変数に設定してください。")
        return

    try:
        import discord
    except ImportError:
        logger.error("discord.py がインストールされていません: pip install discord.py")
        return

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[Bot] ログイン: {client.user}", flush=True)

    @client.event
    async def on_message(message):
        if message.author.bot:
            return

        content = message.content.strip()

        # !予想 [開催場]
        if content.startswith("!予想"):
            args = content.replace("!予想", "").strip()
            cache = _load_cache()
            sep = "━" * 20

            if args:
                # 開催場指定
                entries = _today_entries(cache, venue=args)
                if not entries:
                    await message.channel.send(f"📭 {args}の本日の予想はありません。")
                    return
                header = f"🏟️ {args}の本日の予想"
                lines = [header, sep]
                for _, entry in entries:
                    lines.append(_format_prediction(entry))
                    lines.append(sep)
                await _send_long(message.channel, "\n".join(lines))
            else:
                # 全特別レース
                entries = _today_entries(cache)
                special = [(rid, e) for rid, e in entries
                           if any(kw in e.get("race_name", "") for kw in _SPECIAL_KEYWORDS)]
                if not special:
                    await message.channel.send("📭 本日の特別レース予想はありません。")
                    return
                header = f"🏟️ 本日の特別レース予想（{len(special)}件）"
                lines = [header, sep]
                for _, entry in special:
                    venue = entry.get("venue", "")
                    if venue:
                        lines.append(f"📍 {venue}")
                    lines.append(_format_prediction(entry))
                    lines.append(sep)
                await _send_long(message.channel, "\n".join(lines))

        # !結果
        elif content.startswith("!結果"):
            text = _format_result_summary()
            await _send_long(message.channel, text)

    async def _send_long(channel, text: str):
        """2000文字超を分割送信。"""
        chunks = [text[i:i+1900] for i in range(0, len(text), 1900)]
        for chunk in chunks:
            await channel.send(chunk)

    client.run(token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_bot()
