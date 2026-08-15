# -*- coding: utf-8 -*-
"""
日経225銘柄の中から「本日時点の値下がりランキング」を自分で計算し、
それぞれの業種・時価総額・現在株価・学習済みモデルによる反転確率を
まとめた表を作るスクリプト。

【設計方針】
外部サイトの「値下がりランキングページ」をスクレイピングするのではなく、
reversal_predictor.py と同じ yfinance のデータから自分でランキングを
計算する。これにより、
  - サイトの利用規約違反リスクがない
  - サイトのHTML構造変化で壊れることがない
  - モデル学習時と全く同じ特徴量ロジックを使えるので整合性が保てる
というメリットがある。

【事前準備】
    pip install yfinance pandas numpy scikit-learn joblib
    reversal_predictor.py を実行し、./output/reversal_model.joblib を
    作成しておくこと(このスクリプトはそのモデルを読み込んで使う)。

【使い方】
    python daily_decline_ranking.py
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import requests
import matplotlib
matplotlib.use("Agg")  # GitHub Actions等、画面のない環境でも動くように
import matplotlib.pyplot as plt

from reversal_predictor import add_technical_features, FEATURE_COLUMNS

warnings.filterwarnings("ignore")

JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"


def get_prime_market_info() -> pd.DataFrame:
    """JPX公式の東証上場銘柄一覧からプライム市場銘柄の コード/銘柄名/業種(日本語) を取得する"""
    df = pd.read_excel(JPX_LIST_URL)
    prime = df[df["市場・商品区分"] == "プライム（内国株式）"].copy()
    prime["コード"] = prime["コード"].astype(str).str.zfill(4)
    prime["ticker"] = prime["コード"] + ".T"
    return prime[["コード", "ticker", "銘柄名", "33業種区分"]].reset_index(drop=True)


# =====================================================================
# 設定
# =====================================================================

MODEL_PATH = "./output/reversal_model.joblib"
TOP_N = 20               # 表・CSVに残す件数(値下がり上位N件)
LINE_SEND_N = 20         # LINEに送る件数(反転確率が高い順)
LINE_CHUNK_SIZE = 10     # 1通あたりのバブル数(Flexメッセージは1通最大12バブルまでのため)
HISTORY_PERIOD = "6mo"   # 特徴量計算に必要な過去データの取得期間
CHART_PERIOD = "5y"      # チャート画像の期間(中〜長期)
CHART_DIR = "charts"     # チャート画像の保存先(リポジトリ直下のフォルダ)

# チャート画像を外部(LINE)から見えるURLに変換するための情報。
# GitHub Actions実行時は自動でリポジトリ名・ブランチ名が環境変数から入る。
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "your-account/your-repo")
GITHUB_REF_NAME = os.environ.get("GITHUB_REF_NAME", "main")
RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{GITHUB_REF_NAME}"

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")


# =====================================================================
# 1. 全銘柄の直近データをまとめて取得(1回のAPI呼び出しで済ませる)
# =====================================================================

def fetch_all_latest_data(tickers: list) -> dict:
    """
    複数銘柄の株価データを1回のyf.downloadでまとめて取得する。
    (1銘柄ずつ取得するより高速で、通信エラーも起きにくい)
    戻り値: {ticker: DataFrame} の辞書
    """
    print(f"{len(tickers)}銘柄のデータをまとめて取得中...")
    raw = yf.download(
        tickers=tickers,
        period=HISTORY_PERIOD,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    data = {}
    for ticker in tickers:
        try:
            df = raw[ticker].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except KeyError:
            continue  # その銘柄のデータが取得できなかった場合はスキップ

        if df.empty or df["Close"].dropna().empty:
            continue

        df = df.rename(columns=str.lower)
        df = df.loc[:, ~df.columns.duplicated()]
        df["ticker"] = ticker
        data[ticker] = df

    print(f"  -> {len(data)}銘柄のデータ取得に成功")
    return data


# =====================================================================
# 2. 本日時点の値下がりランキングを作る
# =====================================================================

def build_decline_ranking(data: dict, top_n: int) -> pd.DataFrame:
    """
    各銘柄の最新日の値下がり率を計算し、下落率が大きい順に並べる。
    同時に、反転確率モデルに必要な特徴量も計算しておく。
    """
    rows = []
    for ticker, df in data.items():
        feat = add_technical_features(df)
        latest = feat.iloc[-1]

        if pd.isna(latest["return_1d"]):
            continue

        row = {"ticker": ticker, "latest_date": feat.index[-1], "close": latest["close"]}
        row["decline_return_1d"] = latest["return_1d"]  # 学習時の特徴量名に合わせる
        for col in FEATURE_COLUMNS:
            if col == "decline_return_1d":
                continue
            row[col] = latest[col]
        rows.append(row)

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("decline_return_1d").head(top_n).reset_index(drop=True)
    return ranking


# =====================================================================
# 3. 業種・時価総額などの企業情報を追加
# =====================================================================

def attach_company_info(ranking: pd.DataFrame, jpx_info: pd.DataFrame) -> pd.DataFrame:
    """
    JPXの一覧から日本語の銘柄名・業種をマージし、
    時価総額のみランキング上位銘柄に絞ってyfinanceから取得する。
    """
    ranking = ranking.merge(jpx_info, on="ticker", how="left")

    market_caps = []
    for ticker in ranking["ticker"]:
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        market_caps.append(info.get("marketCap"))

    ranking["market_cap"] = market_caps
    return ranking


# =====================================================================
# 4. 学習済みモデルで反転確率を計算
# =====================================================================

def attach_reversal_probability(ranking: pd.DataFrame, model_path: str) -> pd.DataFrame:
    saved = joblib.load(model_path)
    model = saved["model"] if isinstance(saved, dict) else saved

    X = ranking[FEATURE_COLUMNS].copy()
    valid_mask = X.notna().all(axis=1)

    ranking["reversal_probability"] = np.nan
    if valid_mask.any():
        proba = model.predict_proba(X[valid_mask])[:, 1]
        ranking.loc[valid_mask, "reversal_probability"] = proba

    return ranking


# =====================================================================
# 5. 表として見やすく整形
# =====================================================================

def format_output_table(ranking: pd.DataFrame) -> pd.DataFrame:
    table = ranking.copy()
    table["値下がり率"] = (table["decline_return_1d"] * 100).round(2).astype(str) + "%"
    table["現在株価"] = table["close"].round(1)
    table["時価総額(億円)"] = (table["market_cap"] / 1e8).round(0)
    table["反転確率"] = (table["reversal_probability"] * 100).round(1).astype(str) + "%"

    table = table.rename(columns={
        "コード": "銘柄コード",
        "銘柄名": "企業名",
        "33業種区分": "業種",
    })

    return table[["銘柄コード", "企業名", "業種", "時価総額(億円)", "値下がり率", "現在株価", "反転確率"]]


# =====================================================================
# 6. チャート画像を作成する(5年足 + 現在株価の横線)
# =====================================================================

def generate_chart_images(ranking: pd.DataFrame, output_dir: str = CHART_DIR) -> dict:
    """
    ランキング上位銘柄それぞれについて、中〜長期(5年)の株価チャートを作成し、
    現在株価の位置に横線を引いた画像として保存する。
    戻り値: {ticker: 画像ファイルパス} の辞書
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = {}

    for _, row in ranking.iterrows():
        ticker = row["ticker"]
        current_price = row["close"]

        try:
            hist = yf.download(ticker, period=CHART_PERIOD, auto_adjust=True, progress=False)
        except Exception as e:
            print(f"  [警告] {ticker} のチャート取得に失敗: {e}")
            continue
        if hist.empty:
            print(f"  [警告] {ticker} のチャートデータが空です")
            continue
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist.index, hist["Close"], color="#1f77b4", linewidth=1.2)
        ax.axhline(current_price, color="red", linestyle="--", linewidth=1,
                   label=f"現在株価 {current_price:.0f}円")
        ax.set_title(f"{ticker}  {CHART_PERIOD}チャート")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        code = ticker.replace(".T", "")
        path = os.path.join(output_dir, f"{code}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)

        image_paths[ticker] = path

    return image_paths


# =====================================================================
# 7. LINE Flexメッセージ(カルーセル)を組み立てる
# =====================================================================

def _flex_row(label: str, value: str) -> dict:
    return {
        "type": "box",
        "layout": "baseline",
        "contents": [
            {"type": "text", "text": label, "size": "sm", "color": "#aaaaaa", "flex": 2},
            {"type": "text", "text": str(value), "size": "sm", "flex": 3, "wrap": True},
        ],
    }


def build_flex_carousel(ranking_chunk: pd.DataFrame, image_paths: dict, alt_text: str = "本日の値下がりランキング") -> dict:
    """
    渡されたranking_chunk(最大10件程度を想定)を、1銘柄=1バブル(カード)として
    カルーセルFlexメッセージにする。
    画像はGitHubにpush済みである前提で、raw.githubusercontent.comのURLを組み立てる。
    """
    bubbles = []
    for _, row in ranking_chunk.iterrows():
        ticker = row["ticker"]
        if ticker not in image_paths:
            continue

        image_url = f"{RAW_BASE_URL}/{image_paths[ticker]}"
        market_cap = row.get("market_cap")
        market_cap_text = f"{market_cap / 1e8:.0f}億円" if pd.notna(market_cap) else "不明"
        proba = row.get("reversal_probability")
        proba_text = f"{proba * 100:.1f}%" if pd.notna(proba) else "算出不可"

        bubble = {
            "type": "bubble",
            "hero": {
                "type": "image",
                "url": image_url,
                "size": "full",
                "aspectRatio": "20:13",
                "aspectMode": "cover",
            },
            "body": {
                "type": "box",
                "layout": "vertical",
                "contents": [
                    {
                        "type": "text",
                        "text": f"{row.get('銘柄名', row.get('コード', ticker))} ({row['コード']})",
                        "weight": "bold",
                        "size": "md",
                        "wrap": True,
                    },
                    {
                        "type": "box",
                        "layout": "vertical",
                        "margin": "md",
                        "spacing": "sm",
                        "contents": [
                            _flex_row("業種", row.get("33業種区分", "-")),
                            _flex_row("時価総額", market_cap_text),
                            _flex_row("値下がり率", f"{row['decline_return_1d'] * 100:.2f}%"),
                            _flex_row("現在株価", f"{row['close']:.0f}円"),
                            _flex_row("反転確率", proba_text),
                        ],
                    },
                ],
            },
        }
        bubbles.append(bubble)

    return {
        "type": "flex",
        "altText": alt_text,
        "contents": {"type": "carousel", "contents": bubbles},
    }


# =====================================================================
# 8. LINEへブロードキャスト送信する(友だち全員へ。userIdの取得が不要)
# =====================================================================

def send_line_broadcast(flex_message: dict, channel_access_token: str) -> None:
    url = "https://api.line.me/v2/bot/message/broadcast"
    headers = {
        "Authorization": f"Bearer {channel_access_token}",
        "Content-Type": "application/json",
    }
    payload = {"messages": [flex_message]}

    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    if resp.status_code != 200:
        raise RuntimeError(f"LINE送信に失敗しました: {resp.status_code} {resp.text}")
    print("LINEへの送信に成功しました。")


# =====================================================================
# メイン処理
# =====================================================================

if __name__ == "__main__":
    jpx_info = get_prime_market_info()
    tickers = jpx_info["ticker"].tolist()

    data = fetch_all_latest_data(tickers)
    ranking = build_decline_ranking(data, TOP_N)
    ranking = attach_company_info(ranking, jpx_info)
    ranking = attach_reversal_probability(ranking, MODEL_PATH)

    table = format_output_table(ranking)
    print(f"\n=== 値下がりランキング TOP {TOP_N}(反転確率つき) ===")
    print(table.to_string(index=False))

    os.makedirs("./output", exist_ok=True)
    table.to_csv("./output/daily_decline_ranking.csv", index=False, encoding="utf-8-sig")
    print("\n結果を保存しました: ./output/daily_decline_ranking.csv")

    # LINE送信用: 反転確率が高い順に上位LINE_SEND_N件を選び、チャート画像を作成
    ranking_for_line = ranking.sort_values("reversal_probability", ascending=False).head(LINE_SEND_N).reset_index(drop=True)
    image_paths = generate_chart_images(ranking_for_line)

    # LINE_CHUNK_SIZE件ずつに分割し、それぞれ別のFlexメッセージ(JSON)として保存
    # (実際の送信は、画像をGitHubにpushした後の別ステップ=send_line.pyで行う)
    n_chunks = -(-len(ranking_for_line) // LINE_CHUNK_SIZE)  # 切り上げ除算
    for i in range(n_chunks):
        chunk = ranking_for_line.iloc[i * LINE_CHUNK_SIZE: (i + 1) * LINE_CHUNK_SIZE]
        alt_text = f"本日の反転期待ランキング ({i + 1}/{n_chunks})"
        flex_message = build_flex_carousel(chunk, image_paths, alt_text=alt_text)

        out_path = f"./output/line_flex_message_{i + 1}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(flex_message, f, ensure_ascii=False, indent=2)
        print(f"Flexメッセージを保存しました: {out_path}")
