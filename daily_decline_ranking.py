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

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

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
TOP_N = 20              # ランキングの表示件数(値下がり上位N件)
HISTORY_PERIOD = "6mo"  # 特徴量計算に必要な過去データの取得期間


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

    import os
    os.makedirs("./output", exist_ok=True)
    table.to_csv("./output/daily_decline_ranking.csv", index=False, encoding="utf-8-sig")
    print("\n結果を保存しました: ./output/daily_decline_ranking.csv")
