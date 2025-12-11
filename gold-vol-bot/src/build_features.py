from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_PATH = DATA_DIR / "prices_gold.csv"
SENT_PATH = DATA_DIR / "sentiment_daily.csv"
OUT_PATH = DATA_DIR / "features_daily.csv"


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple RSI implementation using rolling means of gains/losses.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_macro_and_silver(start: pd.Timestamp, end: pd.Timestamp, index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Fetch proxies for:
      - DXY via UUP ETF
      - 10Y yield via ^TNX
      - Silver via SLV ETF

    Align them to the given index (gold price dates).
    """

    def get_close_series(df: pd.DataFrame, ticker: str, out_name: str, scale: float = 1.0) -> pd.Series:
        """
        Safely extract a close/adj-close series from a yfinance DataFrame that might:
          - be single-indexed (Open, High, Low, Close, Adj Close, Volume)
          - be multi-indexed (e.g. ('Adj Close', 'UUP'))
        """
        if df.empty:
            raise RuntimeError(f"No data returned from yfinance for {ticker}.")

        if isinstance(df.columns, pd.MultiIndex):
            # Try to pick out the ticker on the last level
            if ticker in df.columns.get_level_values(-1):
                sub = df.xs(ticker, axis=1, level=-1)
            else:
                sub = df.copy()
                sub.columns = df.columns.get_level_values(0)
        else:
            sub = df

        for col in ["Adj Close", "Close"]:
            if col in sub.columns:
                s = sub[col] * scale
                return s.rename(out_name)

        raise RuntimeError(f"Neither 'Adj Close' nor 'Close' found for {ticker} in downloaded data.")

    print(f">>> Fetching macro series from {start.date()} to {end.date()} via yfinance...")

    dxy = yf.download("UUP", start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
    tnx = yf.download("^TNX", start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
    slv = yf.download("SLV", start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)

    dxy_close = get_close_series(dxy, "UUP", "dxy_close", scale=1.0)
    tnx_yield = get_close_series(tnx, "^TNX", "tnx_yield", scale=1.0 / 100.0)  # convert to decimal
    slv_close = get_close_series(slv, "SLV", "slv_close", scale=1.0)

    macro = pd.DataFrame(index=index)
    macro["dxy_close"] = dxy_close.reindex(index).ffill()
    macro["tnx_yield"] = tnx_yield.reindex(index).ffill()
    macro["slv_close"] = slv_close.reindex(index).ffill()

    macro["dxy_ret_5d"] = np.log(macro["dxy_close"] / macro["dxy_close"].shift(5))
    macro["tnx_chg_1d"] = macro["tnx_yield"].diff()

    return macro


def build_features():
    # ---- Load GLD prices ----
    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    prices = prices.sort_values("date")
    prices.set_index("date", inplace=True)

    close = prices["close"]
    open_ = prices["open"]

    # Returns
    prices["ret_1d"] = np.log(close / close.shift(1))
    prices["ret_5d"] = np.log(close / close.shift(5))
    prices["ret_20d"] = np.log(close / close.shift(20))

    # Overnight vs intraday returns
    prices["overnight_ret"] = np.log(open_ / close.shift(1))
    prices["intraday_ret"] = np.log(close / open_)

    # Rolling vol (annualised)
    prices["vol_10"] = prices["ret_1d"].rolling(10).std() * np.sqrt(252)
    prices["vol_20"] = prices["ret_1d"].rolling(20).std() * np.sqrt(252)
    prices["vol_60"] = prices["ret_1d"].rolling(60).std() * np.sqrt(252)

    # ---- Technical indicators on GLD ----
    prices["ma_5"] = close.rolling(5).mean()
    prices["ma_20"] = close.rolling(20).mean()
    prices["ma_60"] = close.rolling(60).mean()

    prices["rsi_14"] = compute_rsi(close, window=14)

    ma20 = prices["ma_20"]
    std20 = close.rolling(20).std()
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20

    prices["bb_upper"] = upper_band
    prices["bb_lower"] = lower_band
    prices["bb_width"] = (upper_band - lower_band) / ma20
    prices["bb_pct"] = (close - lower_band) / (upper_band - lower_band)

    # ---- Macro & silver series (UUP, ^TNX, SLV) ----
    start = prices.index.min()
    end = prices.index.max()
    macro = fetch_macro_and_silver(start, end, prices.index)

    macro["slv_gold_ratio"] = macro["slv_close"] / close
    macro["slv_gold_ratio_chg"] = macro["slv_gold_ratio"].diff()

    df = prices.join(macro, how="left")

    # ---- Load sentiment & sentiment features ----
    sent = pd.read_csv(SENT_PATH, parse_dates=["date"])
    sent = sent.sort_values("date")
    sent.set_index("date", inplace=True)

    df = df.join(sent, how="left")

    # Fill sentiment with zeros when missing
    df[
        [
            "macro_sent_mean",
            "macro_sent_std",
            "macro_sent_count",
            "gold_sent_mean",
            "gold_sent_std",
            "gold_sent_count",
        ]
    ] = df[
        [
            "macro_sent_mean",
            "macro_sent_std",
            "macro_sent_count",
            "gold_sent_mean",
            "gold_sent_std",
            "gold_sent_count",
        ]
    ].fillna(0)

    # Sentiment flags & rolling windows
    df["has_macro_news"] = (df["macro_sent_count"] > 0).astype(int)
    df["has_gold_news"] = (df["gold_sent_count"] > 0).astype(int)
    df["macro_sent_mean_5d"] = df["macro_sent_mean"].rolling(5).mean()
    df["gold_sent_mean_5d"] = df["gold_sent_mean"].rolling(5).mean()

    # ---- Regime flags ----
    df["regime_high_vol"] = (df["vol_20"] > df["vol_20"].median()).astype(int)
    df["regime_dxy_weak"] = (df["dxy_close"] < df["dxy_close"].rolling(100).mean()).astype(int)
    df["regime_fed_easing"] = (df["tnx_yield"].rolling(20).mean().diff() < 0).astype(int)

    # ---- Macro lags ----
    for lag in [1, 3, 5]:
        df[f"dxy_close_lag{lag}"] = df["dxy_close"].shift(lag)
        df[f"tnx_yield_lag{lag}"] = df["tnx_yield"].shift(lag)
        df[f"slv_ratio_lag{lag}"] = df["slv_gold_ratio"].shift(lag)

    # Forward-fill macro to avoid holes on holidays
    macro_cols = [
        "dxy_close",
        "dxy_ret_5d",
        "tnx_yield",
        "tnx_chg_1d",
        "slv_close",
        "slv_gold_ratio",
        "slv_gold_ratio_chg",
        "dxy_close_lag1",
        "dxy_close_lag3",
        "dxy_close_lag5",
        "tnx_yield_lag1",
        "tnx_yield_lag3",
        "tnx_yield_lag5",
        "slv_ratio_lag1",
        "slv_ratio_lag3",
        "slv_ratio_lag5",
    ]
    df[macro_cols] = df[macro_cols].ffill()

    # ---- Targets ----
    # 1-day ahead return (for backtest PnL)
    df["target_ret_1d"] = df["ret_1d"].shift(-1)
    # 5-day ahead directional label (main target)
    df["target_ret_5d"] = np.log(df["close"].shift(-5) / df["close"])
    df["target_up_5d"] = (df["target_ret_5d"] > 0).astype(int)

    # Drop rows with missing key features or labels
    df = df.dropna(
        subset=[
            "ret_1d",
            "ret_5d",
            "ret_20d",
            "overnight_ret",
            "intraday_ret",
            "vol_10",
            "vol_20",
            "vol_60",
            "ma_5",
            "ma_20",
            "ma_60",
            "rsi_14",
            "bb_width",
            "bb_pct",
            "dxy_close",
            "tnx_yield",
            "slv_gold_ratio",
            "target_ret_1d",
            "target_ret_5d",
            "target_up_5d",
        ]
    )

    df.reset_index(inplace=True)
    df.to_csv(OUT_PATH, index=False)

    print("Sample of feature rows:")
    print(df.tail())
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")


def main():
    build_features()


if __name__ == "__main__":
    main()
