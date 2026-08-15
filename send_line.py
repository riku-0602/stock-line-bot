# -*- coding: utf-8 -*-
"""
daily_decline_ranking.py が保存した ./output/line_flex_message_*.json を
すべて読み込み、順番にLINE Messaging APIのブロードキャスト送信で送るスクリプト。

【前提】
- daily_decline_ranking.py を実行済みで ./output/line_flex_message_1.json,
  line_flex_message_2.json ... が存在すること
- チャート画像(./charts/*.png)が既にGitHubへpushされ、
  raw.githubusercontent.com経由で参照可能な状態になっていること
- 環境変数 LINE_CHANNEL_ACCESS_TOKEN が設定されていること

【使い方】
    python send_line.py
"""

import glob
import json
import time

from daily_decline_ranking import send_line_broadcast, LINE_CHANNEL_ACCESS_TOKEN

if __name__ == "__main__":
    if not LINE_CHANNEL_ACCESS_TOKEN:
        raise RuntimeError("環境変数 LINE_CHANNEL_ACCESS_TOKEN が設定されていません。")

    paths = sorted(glob.glob("./output/line_flex_message_*.json"))
    if not paths:
        raise FileNotFoundError(
            "./output/line_flex_message_*.json が見つかりません。先に daily_decline_ranking.py を実行してください。"
        )

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            flex_message = json.load(f)

        print(f"{path} を送信中...")
        send_line_broadcast(flex_message, LINE_CHANNEL_ACCESS_TOKEN)
        time.sleep(1)  # 連続送信によるレート制限を避けるための間隔
