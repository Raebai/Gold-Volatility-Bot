# Gold-Volatility-Bot

A daily research pipeline that scores next-day direction in gold (the GLD ETF) from price, volatility and news-sentiment features, and logs every prediction so the strategy can be judged after the fact.

## Why

Gold moves on macro news as much as on price action, but the two are usually studied separately — a chart on one screen, a newsfeed on the other. This joins them: it pulls ten years of GLD prices and a rolling window of macro and gold-related headlines, turns both into a single daily feature row, and asks one question — is tomorrow up or down?

The point is the logging. Every prediction is appended with its date and probability, so the record accumulates honestly instead of being re-fit after the fact.

## How it works

`run_daily.py` runs five stages in order:

1. **Prices** — `fetch_prices.py` downloads ten years of daily GLD OHLCV from Yahoo Finance.
2. **News** — `fetch_news.py` pulls macro and gold-related headlines from Alpha Vantage's news-sentiment endpoint.
3. **Sentiment** — `aggregate_sentiment.py` collapses those headlines into one daily sentiment score per topic.
4. **Features** — `build_features.py` joins the two and derives returns (1/5/20-day), overnight and intraday moves, realised volatility (10/20/60-day), RSI, and rolling sentiment with event flags.
5. **Model** — `train_and_predict.py` trains a calibrated XGBoost classifier, reports accuracy and ROC-AUC on a held-out split, prints today's signal, and appends to `data/predictions_log.csv` and `data/backtest_log_validation.csv`.

`src/analysis/strategy_analysis.ipynb` is where results are actually examined — hit rate, volatility regime, and whether the edge survives costs.

## Stack

Python · pandas · NumPy · scikit-learn · XGBoost · statsmodels · `arch` (GARCH) · yfinance · Alpha Vantage API · Jupyter

## Running it

```bash
git clone https://github.com/Raebai/Gold-Volatility-Bot.git
cd Gold-Volatility-Bot/gold-vol-bot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then add your free Alpha Vantage key
python src/run_daily.py
```

A free Alpha Vantage key is required — `fetch_news.py` raises immediately without `ALPHAVANTAGE_API_KEY`.

## Status

Works end to end: run it and you get a signal for today, appended to the log.

What's rough, honestly:

- **It is a research pipeline, not a trading system.** No broker, no execution, no position sizing, and no transaction costs in the headline accuracy figure.
- **The model retrains on every run** rather than loading a saved artefact, so results shift slightly day to day.
- **Nothing schedules it.** "Daily" describes intent; you run it yourself.
- Alpha Vantage's free tier is rate-limited, which caps how much news history can be rebuilt in one sitting.
