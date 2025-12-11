from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
NEWS_PATH = DATA_DIR / "news_raw.csv"
OUT_PATH = DATA_DIR / "sentiment_daily.csv"


def aggregate_daily_sentiment():
    if not NEWS_PATH.exists():
        raise RuntimeError("news_raw.csv not found. Run fetch_news.py first.")

    df = pd.read_csv(NEWS_PATH, parse_dates=["published_at"])
    if df.empty:
        raise RuntimeError("No news data found to aggregate (news_raw.csv is empty).")

    # Use date (no time) as key for daily aggregation
    df["date"] = df["published_at"].dt.date

    # ---------- 1) MACRO SENTIMENT: all articles ----------
    macro_grouped = df.groupby("date").agg(
        macro_sent_mean=("overall_sentiment_score", "mean"),
        macro_sent_std=("overall_sentiment_score", "std"),
        macro_sent_count=("overall_sentiment_score", "count"),
    )

    # ---------- 2) GOLD SENTIMENT: only articles mentioning gold/XAU ----------
    text = (df["title"].fillna("") + " " + df["summary"].fillna("")).str.lower()
    gold_mask = text.str.contains("gold") | text.str.contains("xau")

    gold_df = df[gold_mask].copy()

    if gold_df.empty:
        # No explicit gold news -> create empty gold columns with zeros
        gold_grouped = macro_grouped.copy()
        gold_grouped["gold_sent_mean"] = 0.0
        gold_grouped["gold_sent_std"] = 0.0
        gold_grouped["gold_sent_count"] = 0
    else:
        gold_grouped = gold_df.groupby("date").agg(
            gold_sent_mean=("overall_sentiment_score", "mean"),
            gold_sent_std=("overall_sentiment_score", "std"),
            gold_sent_count=("overall_sentiment_score", "count"),
        )

    # Join macro + gold on date
    grouped = macro_grouped.join(gold_grouped, how="left")

    # Fill any missing gold columns where no gold news that day
    grouped[["gold_sent_mean", "gold_sent_std", "gold_sent_count"]] = grouped[
        ["gold_sent_mean", "gold_sent_std", "gold_sent_count"]
    ].fillna(0)

    grouped = grouped.reset_index()  # bring date back as a column
    grouped.to_csv(OUT_PATH, index=False)

    print("Sample of aggregated daily sentiment:")
    print(grouped.tail())
    print(f"\nSaved {len(grouped)} rows to {OUT_PATH}")


def main():
    aggregate_daily_sentiment()


if __name__ == "__main__":
    main()
