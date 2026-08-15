# -*- coding: utf-8 -*-
"""
daily_decline_ranking.py が保存した ./output/line_flex_message.json を読み込み、
LINE Messaging APIのブロードキャスト送信でLINEへ送るだけのスクリプト。

【前提】
- daily_decline_ranking.py を実行済みで ./output/line_flex_message.json が存在すること
- チャート画像(./charts/*.png)が既にGitHubへpushされ、
  raw.githubusercontent.com経由で参照可能な状態になっていること
  (画像がpushされる前に送信すると、LINE側で画像が表示されない)
- 環境変数 LINE_CHANNEL_ACCESS_TOKEN が設定されていること
  (GitHub Actionsでは secrets.LINE_CHANNEL_ACCESS_TOKEN を渡す)

【使い方】
    python send_line.py
"""

import json
import os

from daily_decline_ranking import send_line_broadcast, LINE_CHANNEL_ACCESS_TOKEN

if __name__ == "__main__":
    if not LINE_CHANNEL_ACCESS_TOKEN:
        raise RuntimeError("環境変数 LINE_CHANNEL_ACCESS_TOKEN が設定されていません。")

    path = "./output/line_flex_message.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} が見つかりません。先に daily_decline_ranking.py を実行してください。")

    with open(path, "r", encoding="utf-8") as f:
        flex_message = json.load(f)

    send_line_broadcast(flex_message, LINE_CHANNEL_ACCESS_TOKEN)
