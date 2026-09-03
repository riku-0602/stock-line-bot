# -*- coding: utf-8 -*-
"""
株価「値下がりランキング」銘柄の反転(リバウンド)確率を
機械学習で予測するための学習用スクリプト。

【全体の流れ】
1. yfinanceで対象銘柄の日足データを取得
2. 各銘柄・各日について「大きく値下がりした日(イベント日)」を抽出
3. イベント日時点での特徴量(値下がり率、出来高、RSI、移動平均乖離など)を作成
4. イベント日からN営業日後に一定以上値上がりしたか(反転成功)をラベル付け
5. 時系列を考慮して学習データ/検証データに分割し、RandomForestで学習
6. 学習済みモデルで「反転確率」を出力できるようにする

【事前準備(すべて無料)】
    pip install yfinance pandas numpy scikit-learn joblib

【注意】
- yfinanceは無料ですが、大量銘柄×長期間のダウンロードは時間がかかります。
  まずは対象銘柄(TICKERS)を少数に絞ってテストしてください。
- 日本株はyfinance上で証券コードに ".T" を付けて指定します(例: トヨタ = 7203.T)。
- ここでは「セクター/時価総額」もyfinanceのTicker.infoから取得しますが、
  情報が欠落している銘柄もあるため、Noneになる場合があります。
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

warnings.filterwarnings("ignore")


# =====================================================================
# 1. 設定(ここを自分の目的に合わせて調整してください)
# =====================================================================

# 日経225 構成銘柄コード(2024年時点ベース。入れ替えにより一部が最新でない可能性があるため、
# 上場廃止・銘柄コード変更などのエラーが出た場合はそのコードをリストから除外してください)
NIKKEI225_CODES = (
    "6857","8267","5201","2802","6770","6113","9202","8304","2502","3407",
    "4503","7832","5108","7751","6952","9022","9502","4519","7762","1721",
    "7186","8253","4751","7912","8750","4568","6367","1925","8601","2432",
    "4061","6902","4324","4631","5714","9020","6361","4523","5020","6954",
    "9983","6504","4901","5803","6702","8354","5801","6674","1808","7205",
    "6305","7004","6501","7267","7741","5019","7013","1605","3099","7202",
    "8001","3086","9201","8697","6178","2914","5411","1963","6473","1812",
    "4452","7012","9107","9433","9008","9009","6861","2801","2503","5406",
    "6301","9766","4902","6326","3405","6971","4151","6920","4689","2413",
    "8002","8252","7261","2269","4385","6479","4188","8058","6503","8802",
    "7011","9301","5711","7211","8306","8031","4183","8801","5706","9104",
    "8411","8725","6981","6701","3659","5333","2282","2871","6594","7731",
    "7974","5214","9147","3863","5401","9432","9101","4021","7201","2002",
    "1332","9843","6988","8604","6471","6472","9613","1802","9007","3861",
    "6103","7733","6645","4661","8591","9532","4578","5541","6752","4755",
    "6098","6723","8308","4004","7752","2501","7735","9735","6724","1928",
    "3382","6753","1803","4063","4507","4911","5831","6273","9434","9984",
    "2768","8630","6758","7270","3436","4005","8053","5802","6302","5713",
    "8316","8309","5232","4506","8830","7269","8795","5233","1801","6976",
    "2531","8233","4502","6762","3401","4543","8331","5631","9503","5101",
    "9001","9602","5301","8766","4043","9501","8035","9531","8804","9005",
    "3289","7911","3402","4042","5332","7203","8015","4704","4208","9021",
    "7951","7272","9064","6506","6841",
)

JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"


def get_prime_market_codes() -> list:
    """JPX公式の東証上場銘柄一覧から、プライム市場銘柄の証券コード一覧を取得する"""
    df = pd.read_excel(JPX_LIST_URL)
    prime = df[df["市場・商品区分"] == "プライム（内国株式）"]
    return prime["コード"].astype(str).str.zfill(4).tolist()


# =====================================================================
# 1. 設定(ここを自分の目的に合わせて調整してください)
# =====================================================================

@dataclass
class Config:
    # 対象銘柄(mainではプライム市場全体に差し替える。日経225はデフォルトのフォールバック)
    tickers: tuple = tuple(f"{code}.T" for code in NIKKEI225_CODES)

    start_date: str = "2015-01-01"
    end_date: str = None  # Noneなら現在まで

    # 「値下がりイベント」の定義:1日の下落率がこの値以下(例: -5%)ならイベントとする
    decline_threshold: float = -0.05

    # 反転判定に使う保有期間(営業日数)
    forward_days: int = 5

    # この保有期間内の最大上昇率がこの値以上なら「反転成功」とラベル付け
    reversal_threshold: float = 0.10

    # TimeSeriesSplitでの交差検証に使う分割数
    n_splits: int = 5

    model_out_path: str = "./output/reversal_model.joblib"


# CFGはモジュール読み込み時には作らず、__main__内でプライム市場銘柄を取得してから生成する
# (他のスクリプトがNIKKEI225_CODESだけを使う場合に、不要な通信が走らないようにするため)


# =====================================================================
# 2. データ取得
# =====================================================================

def fetch_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """1銘柄分の日足OHLCVを取得する"""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df

    # 最近のyfinanceは1銘柄指定でも列が2階層(MultiIndex)になることがあるため、
    # その場合は1階層(例: 'Close', 'Volume' など)に整形する
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)

    # 列名が重複していると後の計算でDataFrameが返ってしまうため、重複列を除去
    df = df.loc[:, ~df.columns.duplicated()]

    df["ticker"] = ticker
    return df


def fetch_meta(ticker: str) -> dict:
    """銘柄の業種・時価総額などの静的情報を取得する(取得できない場合はNone)"""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "long_name": info.get("longName"),
    }


# =====================================================================
# 3. 特徴量作成
# =====================================================================

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI・移動平均乖離・出来高比率などのテクニカル特徴量を追加する"""
    df = df.copy()
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    # 移動平均と乖離率
    df["ma25"] = df["close"].rolling(25).mean()
    df["ma75"] = df["close"].rolling(75).mean()
    df["dev_ma25"] = (df["close"] - df["ma25"]) / df["ma25"]
    df["dev_ma75"] = (df["close"] - df["ma75"]) / df["ma75"]

    # 出来高の20日平均に対する比率(急落時の出来高急増を捉える)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # RSI(14日)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ボラティリティ(20日リターンの標準偏差)
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    return df


# =====================================================================
# 4. イベント抽出 + ラベル作成
# =====================================================================

def build_events(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    値下がりイベント日を抽出し、特徴量とラベル(反転成功=1/失敗=0)を付与する。
    ラベルは「イベント翌日からforward_days営業日以内の終値の最大上昇率」が
    reversal_threshold以上かどうかで判定する(未来のデータをのぞき見しないよう、
    イベント日当日の情報までしか特徴量には使わない)。
    """
    df = df.reset_index(drop=False)
    events = []

    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row["return_1d"]) or row["return_1d"] > cfg.decline_threshold:
            continue  # 閾値ほど下落していない日はスキップ

        # forward_days先までのデータが存在するかチェック
        future_slice = df.iloc[i + 1: i + 1 + cfg.forward_days]
        if len(future_slice) < cfg.forward_days:
            continue  # 将来データが足りない(直近すぎる)イベントは除外

        base_price = row["close"]
        max_future_return = (future_slice["close"].max() - base_price) / base_price
        label = 1 if max_future_return >= cfg.reversal_threshold else 0

        events.append({
            "date": row["Date"] if "Date" in df.columns else row.get("index"),
            "ticker": row["ticker"],
            "close": row["close"],
            "decline_return_1d": row["return_1d"],
            "return_5d": row["return_5d"],
            "return_20d": row["return_20d"],
            "dev_ma25": row["dev_ma25"],
            "dev_ma75": row["dev_ma75"],
            "vol_ratio": row["vol_ratio"],
            "rsi14": row["rsi14"],
            "volatility_20d": row["volatility_20d"],
            "max_future_return": max_future_return,
            "label": label,
        })

    return pd.DataFrame(events)


# =====================================================================
# 5. データセット構築(全銘柄まとめて)
# =====================================================================

def build_dataset(cfg: Config) -> pd.DataFrame:
    all_events = []
    unique_tickers = list(dict.fromkeys(cfg.tickers))  # 順序を保ったまま重複除去
    for ticker in unique_tickers:
        print(f"[取得中] {ticker}")
        raw = fetch_price_data(ticker, cfg.start_date, cfg.end_date)
        if raw.empty or len(raw) < 100:
            print(f"  -> データ不足のためスキップ: {ticker}")
            continue
        feat = add_technical_features(raw)
        events = build_events(feat, cfg)
        if not events.empty:
            all_events.append(events)
        time.sleep(0.5)  # yfinanceへの過剰アクセス防止

    if not all_events:
        raise RuntimeError("イベントが1件も抽出できませんでした。閾値や期間を見直してください。")

    dataset = pd.concat(all_events, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    dataset = dataset.sort_values("date").reset_index(drop=True)
    return dataset


# =====================================================================
# 6. 学習・評価
# =====================================================================

FEATURE_COLUMNS = [
    "decline_return_1d",
    "return_5d",
    "return_20d",
    "dev_ma25",
    "dev_ma75",
    "vol_ratio",
    "rsi14",
    "volatility_20d",
]


def tune_random_forest(dataset: pd.DataFrame, cfg: Config):
    """
    RandomForestのハイパーパラメータをRandomizedSearchCVで探索する。
    交差検証にはTimeSeriesSplit(未来のデータで過去を学習しないよう時系列順を保つ)を使い、
    評価指標はAUC(反転する/しないの分離性能)とする。
    探索後、最良パラメータで全データを使い最終モデルを再学習して保存する。
    """
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["label"]).sort_values("date").reset_index(drop=True)
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    print(f"\n学習に使うイベント総数: {len(X)}  (陽性率: {y.mean():.1%})")
    if len(X) < cfg.n_splits * 30:
        print(f"警告: データ件数に対してn_splits={cfg.n_splits}は多い可能性があります。"
              f"n_splitsを減らすか、tickers/decline_thresholdを見直してください。")

    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)

    # 探索するハイパーパラメータの範囲
    param_distributions = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8, 10, None],
        "min_samples_leaf": [1, 3, 5, 10, 20, 30],
        "min_samples_split": [2, 5, 10, 20],
        "max_features": ["sqrt", "log2", 0.5, None],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=40,                # 40通りのパラメータ組み合わせを試す(多いほど丁寧だが遅くなる)
        scoring="roc_auc",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        refit=True,                # 探索後、全データで自動的に再学習される
        verbose=1,
    )
    search.fit(X, y)

    print("\n=== ハイパーパラメータ探索結果 ===")
    print(f"最良パラメータ: {search.best_params_}")
    print(f"交差検証での平均AUC: {search.best_score_:.3f}")

    # 最良パラメータでのfold別スコアも個別に確認する(安定性のチェック)
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = RandomForestClassifier(**search.best_params_, random_state=42, n_jobs=-1)
        fold_model.fit(X_train, y_train)
        proba = fold_model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = np.nan

        fold_records.append({
            "fold": fold, "n_train": len(train_idx), "n_test": len(test_idx),
            "accuracy": acc, "auc": auc,
        })

    fold_df = pd.DataFrame(fold_records)
    print("\n=== fold別の最終評価(最良パラメータ) ===")
    print(fold_df.to_string(index=False))
    print(f"\n【最終評価】平均AUC: {fold_df['auc'].mean():.3f} (±{fold_df['auc'].std():.3f})"
          f" / 平均Accuracy: {fold_df['accuracy'].mean():.3f} (±{fold_df['accuracy'].std():.3f})")

    final_model = search.best_estimator_  # 全データで再学習済み(refit=Trueのため)

    importance = pd.Series(final_model.feature_importances_, index=FEATURE_COLUMNS)
    print("\n=== 特徴量重要度(最終モデル) ===")
    print(importance.sort_values(ascending=False))

    joblib.dump(
        {
            "model": final_model,
            "best_params": search.best_params_,
            "cv_best_auc": search.best_score_,
            "fold_results": fold_df,
        },
        cfg.model_out_path,
    )
    print(f"\nモデルを保存しました: {cfg.model_out_path}")

    return final_model, fold_df


# =====================================================================
# 7. 新規の値下がり銘柄に対して確率を予測する関数(実運用イメージ)
# =====================================================================

def predict_reversal_probability(model, latest_feature_row: dict) -> float:
    """
    最新の値下がり銘柄1件分の特徴量(dict)を渡すと、反転確率(0〜1)を返す。
    latest_feature_row は FEATURE_COLUMNS と同じキーを持つ辞書。
    model には tune_random_forest() が返した final_model、
    または joblib.load() で読み込んだ辞書の ["model"] を渡す。
    """
    x = pd.DataFrame([latest_feature_row])[FEATURE_COLUMNS]
    proba = model.predict_proba(x)[0, 1]
    return float(proba)


# =====================================================================
# メイン処理
# =====================================================================

if __name__ == "__main__":
    import os
    os.makedirs("./output", exist_ok=True)  # 保存先フォルダが無ければ作成

    print("プライム市場銘柄一覧を取得中...")
    prime_codes = get_prime_market_codes()
    CFG = Config(tickers=tuple(f"{code}.T" for code in prime_codes))
    print(f"対象銘柄数: {len(CFG.tickers)}")

    dataset = build_dataset(CFG)
    print(f"\n抽出された値下がりイベント総数: {len(dataset)}")
    dataset.to_csv("./output/reversal_events_dataset.csv", index=False)
    print("イベントデータセットを保存しました: ./output/reversal_events_dataset.csv")

    model, fold_df = tune_random_forest(dataset, CFG)

    # 使用例: 直近の値下がり銘柄の特徴量を渡して確率を出す場合
    # sample_row = {
    #     "decline_return_1d": -0.07,
    #     "return_5d": -0.10,
    #     "return_20d": -0.15,
    #     "dev_ma25": -0.12,
    #     "dev_ma75": -0.08,
    #     "vol_ratio": 2.3,
    #     "rsi14": 25.0,
    #     "volatility_20d": 0.03,
    # }
    # print(predict_reversal_probability(model, sample_row))
