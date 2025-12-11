from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "prices_gold.csv"


def fetch_gld_daily():
    """
    Fetch daily GLD ETF prices from Yahoo Finance using yfinance
    and return a clean DataFrame with columns:
    date, open, high, low, close, volume
    """
    end = date.today()
    start = end - timedelta(days=365 * 10)  # last 10 years

    print(f"Downloading GLD data from {start} to {end} via yfinance...")
    df = yf.download("GLD", start=start, end=end, progress=False)

    if df.empty:
        raise RuntimeError("No data returned from yfinance for GLD.")

    # If columns are multi-index (Price/Ticker), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        # Take just the 'GLD' level if present
        if "GLD" in df.columns.get_level_values(-1):
            df = df.xs("GLD", axis=1, level=-1)
        else:
            # Fallback: drop the top level
            df.columns = df.columns.get_level_values(-1)

    # Keep only needed columns and tidy up
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.reset_index(inplace=True)  # move Date from index to column
    df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df["date"] = df["date"].dt.date  # keep only the date part

    return df


def main():
    df = fetch_gld_daily()

    print("Sample of cleaned data:")
    print(df.tail())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()