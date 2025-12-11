"""
run_daily.py

One-shot pipeline runner:
1) Update GLD prices
2) Fetch latest macro + gold-related news
3) Aggregate daily macro + gold sentiment
4) Rebuild feature dataset
5) Train model and print today's signal
"""

from fetch_prices import main as fetch_prices_main
from fetch_news import main as fetch_news_main
from aggregate_sentiment import main as aggregate_sentiment_main
from build_features import main as build_features_main
from train_and_predict import main as train_and_predict_main


def main():
    print("=== [1/5] Updating GLD prices ===")
    fetch_prices_main()

    print("\n=== [2/5] Fetching macro + gold news ===")
    fetch_news_main()

    print("\n=== [3/5] Aggregating daily sentiment ===")
    aggregate_sentiment_main()

    print("\n=== [4/5] Rebuilding feature dataset ===")
    build_features_main()

    print("\n=== [5/5] Training model & generating today's signal ===")
    train_and_predict_main()

    print("\n✅ Daily pipeline completed.")


if __name__ == "__main__":
    main()
