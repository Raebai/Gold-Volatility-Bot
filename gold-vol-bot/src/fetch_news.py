import os
import requests
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

print(">>> fetch_news.py starting up...")  # DEBUG PRINT

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

if API_KEY is None:
    raise RuntimeError("Missing ALPHAVANTAGE_API_KEY in .env")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_PATH = DATA_DIR / "news_raw.csv"


def fetch_market_news(limit: int = 200) -> pd.DataFrame:
    """
    Fetch recent market/news sentiment from Alpha Vantage.
    """
    url = "https://www.alphavantage.co/query"

    topics = "financial_markets,economy_monetary,economy_macro,finance"

    params = {
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "limit": limit,
        "apikey": API_KEY,
    }

    print("Requesting news from Alpha Vantage...")  # DEBUG PRINT
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    if "Note" in data:
        print("Alpha Vantage NOTE (likely rate limit):")
        print(data["Note"])
        return pd.DataFrame()

    if "Error Message" in data:
        print("Alpha Vantage ERROR:")
        print(data["Error Message"])
        return pd.DataFrame()

    feed = data.get("feed")
    if feed is None:
        print("No 'feed' key in response, raw data:")
        print(data)
        return pd.DataFrame()

    print(f"Total articles in feed: {len(feed)}")

    rows = []
    for item in feed:
        time_published = item.get("time_published")
        if not time_published:
            continue

        try:
            pub_dt = pd.to_datetime(time_published, format="%Y%m%dT%H%M%S", utc=True)
        except Exception:
            pub_dt = pd.to_datetime(time_published, errors="coerce", utc=True)

        if pd.isna(pub_dt):
            continue

        title = item.get("title", "")
        summary = item.get("summary", "")
        overall_sentiment = float(item.get("overall_sentiment_score", 0.0))

        rows.append(
            {
                "published_at": pub_dt,
                "title": title,
                "summary": summary,
                "overall_sentiment_score": overall_sentiment,
                "source": item.get("source"),
                "url": item.get("url"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No articles parsed into DataFrame.")
        return df

    df = df.sort_values("published_at").reset_index(drop=True)
    print(f"Parsed {len(df)} usable articles.")
    return df


def main():
    print(">>> main() in fetch_news.py running...")  # DEBUG PRINT
    df = fetch_market_news(limit=200)
    print("Sample of downloaded news:")
    print(df.tail())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    print(">>> __name__ == '__main__', calling main()")  # DEBUG PRINT
    main()
