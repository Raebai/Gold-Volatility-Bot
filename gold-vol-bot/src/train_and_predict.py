from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

print(">>> train_and_predict.py loaded")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURES_PATH = DATA_DIR / "features_daily.csv"

PRED_LOG_PATH = DATA_DIR / "predictions_log.csv"
BT_LOG_PATH = DATA_DIR / "backtest_log_validation.csv"

# Use the richer feature set built in build_features.py
FEATURE_COLS = [
    # Returns & volatility
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "overnight_ret",
    "intraday_ret",
    "vol_10",
    "vol_20",
    "vol_60",

    # Sentiment (daily + rolling + flags)
    "macro_sent_mean",
    "macro_sent_std",
    "macro_sent_count",
    "macro_sent_mean_5d",
    "gold_sent_mean",
    "gold_sent_std",
    "gold_sent_count",
    "gold_sent_mean_5d",
    "has_macro_news",
    "has_gold_news",

    # Technicals on GLD
    "ma_5",
    "ma_20",
    "ma_60",
    "rsi_14",
    "bb_width",
    "bb_pct",

    # Macro proxies & lags
    "dxy_close",
    "dxy_ret_5d",
    "tnx_yield",
    "tnx_chg_1d",
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

    # Regimes
    "regime_high_vol",
    "regime_dxy_weak",
    "regime_fed_easing",
]

# Main target: 5-day ahead direction (set in build_features.py)
TARGET_COL = "target_up_5d"


def load_dataset() -> pd.DataFrame:
    print(">>> Loading dataset from", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date")

    # Restrict to last ~900 days (modern regime, good sentiment coverage)
    max_date = df["date"].max()
    window_start = max_date - pd.Timedelta(days=900)
    df = df[df["date"] >= window_start]

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "target_ret_1d"])
    print(f">>> Loaded {len(df)} rows after dropping NA (last ~900 days)")
    return df


def build_model() -> XGBClassifier:
    # Reasonable default XGBoost config for noisy market data
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    return model


def position_from_signal(prob_up: float, row: pd.Series) -> float:
    """
    Turn model probability + current market regime into a continuous position in [-1, 1].

    Key ideas:
      - Use an edge threshold so tiny edges => no trade.
      - Scale linearly with edge up to a max conviction.
      - Vol-target the position (~10% ann. vol) using 20d realised vol.
      - Tilt size up/down based on simple macro regimes already in the feature set.
    """
    # ----- Edge & basic signal -----
    edge = float(prob_up) - 0.5
    edge_abs = abs(edge)

    # Minimum edge to take any risk (3% by default)
    EDGE_MIN = 0.03
    EDGE_MAX = 0.20  # 20% edge => full conviction

    if edge_abs < EDGE_MIN:
        return 0.0

    # Scale edge so that EDGE_MAX => raw_signal = +/-1
    raw_signal = np.clip(edge / EDGE_MAX, -1.0, 1.0)

    # ----- Vol targeting -----
    vol_20 = float(row.get("vol_20", np.nan))
    if not np.isfinite(vol_20) or vol_20 <= 0:
        vol_scale = 0.0
    else:
        TARGET_VOL = 0.10  # target 10% annualised
        vol_scale = TARGET_VOL / vol_20
        vol_scale = float(np.clip(vol_scale, 0.5, 1.5))

    pos = raw_signal * vol_scale

    # ----- Regime tilts (small, controlled adjustments) -----
    # These regime flags already exist in the feature set.
    regime_high_vol = int(row.get("regime_high_vol", 0))
    regime_dxy_weak = int(row.get("regime_dxy_weak", 0))
    regime_fed_easing = int(row.get("regime_fed_easing", 0))

    dxy_ret_5d = float(row.get("dxy_ret_5d", 0.0))
    tnx_chg_1d = float(row.get("tnx_chg_1d", 0.0))

    regime_scale = 1.0

    # In structurally high-vol regimes, de-risk a bit.
    if regime_high_vol:
        regime_scale *= 0.7

    # Long gold works better when USD is weak and Fed is easing.
    if edge > 0:
        if regime_dxy_weak:
            regime_scale *= 1.15
        if regime_fed_easing:
            regime_scale *= 1.10
        if dxy_ret_5d > 0:  # dollar recently stronger -> be a bit more cautious
            regime_scale *= 0.9
        if tnx_chg_1d > 0:  # yields popping up -> slightly reduce long
            regime_scale *= 0.9

    # Short gold works better when USD is strong and yields rising.
    if edge < 0:
        if not regime_dxy_weak and dxy_ret_5d > 0:
            regime_scale *= 1.10
        if tnx_chg_1d > 0:
            regime_scale *= 1.10

    # Clip regime scaling to avoid crazy leverage
    regime_scale = float(np.clip(regime_scale, 0.5, 1.5))

    pos *= regime_scale

    # Final safety clip
    pos = float(np.clip(pos, -1.0, 1.0))
    return pos


def train_and_backtest(df: pd.DataFrame):
    print(">>> Training model...")
    max_date = df["date"].max()
    cutoff = max_date - timedelta(days=365)  # last ~1y as validation

    train = df[df["date"] < cutoff].copy()
    valid = df[df["date"] >= cutoff].copy()

    if train.empty or valid.empty:
        print(">>> Not enough data for 1-year validation; using all data as train.")
        train = df.copy()
        valid = pd.DataFrame(columns=df.columns)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    base_model = build_model()
    base_model.fit(X_train, y_train)

    # Feature importance from base model
    importances = base_model.feature_importances_
    fi = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
        .sort_values("importance", ascending=False)
    )
    print("\n>>> Top 15 features by importance:")
    print(fi.head(15).to_string(index=False))

    # Calibrated model for better probabilities
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    # Validation metrics & backtest with transaction costs
    if not valid.empty:
        valid = valid.copy().reset_index(drop=True)
        X_valid = valid[FEATURE_COLS]
        y_valid = valid[TARGET_COL]

        prob_valid = calibrated.predict_proba(X_valid)[:, 1]
        pred_valid = (prob_valid > 0.5).astype(int)
        acc = accuracy_score(y_valid, pred_valid)
        auc = roc_auc_score(y_valid, prob_valid)
        print(f">>> Validation accuracy (5d): {acc:.3f}, AUC: {auc:.3f}")

        target_ret_1d = valid["target_ret_1d"].values

        # ----- Continuous positions with regime-aware logic -----
        positions = []
        for i, row in valid.iterrows():
            p = float(prob_valid[i])
            pos = position_from_signal(p, row)
            positions.append(pos)
        positions = np.array(positions, dtype=float)

        # Transaction costs: proportional to daily turnover
        cost_per_turnover = 0.0005  # 5 bps per round-trip (approx)

        position_prev = np.concatenate([[0.0], positions[:-1]])
        turnover = np.abs(positions - position_prev)
        trading_costs = cost_per_turnover * turnover

        # Daily log P&L: position * next-day log return minus costs
        gross_pnl = positions * target_ret_1d
        net_pnl = gross_pnl - trading_costs

        # Backtest stats
        if np.any(np.isfinite(net_pnl)) and np.nanstd(net_pnl) > 0:
            cum_log_ret = np.nancumsum(net_pnl)
            n = len(net_pnl)
            years = n / 252.0
            ann_return = np.exp(cum_log_ret[-1] / years) - 1
            sharpe = (np.nanmean(net_pnl) / np.nanstd(net_pnl)) * np.sqrt(252)

            equity = np.exp(cum_log_ret)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_dd = float(np.nanmin(dd))

            print(
                f">>> Backtest on validation (net of costs, approx): "
                f"Ann. return {ann_return:.2%}, Sharpe {sharpe:.2f}, Max DD {max_dd:.2%}"
            )

            # ---- Save backtest series for inspection ----
            bt_df = pd.DataFrame(
                {
                    "date": valid["date"],
                    "prob_up": prob_valid,
                    "position": positions,
                    "position_prev": position_prev,
                    "turnover": turnover,
                    "trading_cost": trading_costs,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "equity": equity,
                }
            )
            bt_df.to_csv(BT_LOG_PATH, index=False)
            print(f">>> Saved validation backtest series to {BT_LOG_PATH}")
        else:
            print(">>> Not enough variation in PnL to compute Sharpe.")
    else:
        print(">>> No separate validation set; skipping backtest.")

    return calibrated


def log_prediction(row: pd.Series, prob_up: float, position: float):
    """
    Append today's prediction to a CSV log so you can analyse
    live performance over time (out-of-sample).
    """
    log_row = pd.DataFrame(
        [
            {
                "date": row["date"],
                "prob_up_5d": prob_up,
                "vol_20": row["vol_20"],
                "position": position,
            }
        ]
    )

    if PRED_LOG_PATH.exists():
        existing = pd.read_csv(PRED_LOG_PATH, parse_dates=["date"])
        combined = pd.concat([existing, log_row], ignore_index=True)
        combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
        combined.to_csv(PRED_LOG_PATH, index=False)
    else:
        log_row.to_csv(PRED_LOG_PATH, index=False)

    print(f">>> Logged prediction to {PRED_LOG_PATH}")


def main():
    print(">>> main() in train_and_predict.py starting")
    df = load_dataset()
    model = train_and_backtest(df)

    # Use last row as "today" (note: last 4–5 days are dropped due to 5d target)
    last_row = df.iloc[-1]
    X_last = pd.DataFrame([last_row[FEATURE_COLS].values], columns=FEATURE_COLS)
    prob_up = model.predict_proba(X_last)[0, 1]

    position = position_from_signal(prob_up, last_row)

    vol_20 = float(last_row["vol_20"])

    print("\n--- Today’s signal (5d target model) ---")
    print(f"Date used: {last_row['date'].date()}")
    print(f"Probability GLD (gold proxy) is UP over next 5 days: {prob_up:.2%}")
    print(f"20-day annualised volatility estimate: {vol_20:.2%}")
    print(f"Suggested position (from -1 to +1): {position:.3f}")

    # Log today's prediction
    log_prediction(last_row, prob_up, position)


if __name__ == "__main__":
    print(">>> __name__ == '__main__' in train_and_predict.py, calling main()")
    main()
