import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.simple_features import SimpleFeatureCalculator
from src.models.quantile_regressor import QuantileRegressor
from src.models.decision_engine import DecisionEngine

def main():
    print("=" * 60)
    print("BACKTEST V2: Dollar Basis & Signal Expansion")
    print("=" * 60)

    # 1. Config
    ACCOUNT_SIZE = 10000.0  # $10,000
    LOT_SIZE_USD = 10000.0  # 1 Mini Lot ($10,000 position)
    HORIZON = 4             # 1 hour
    
    # We calibrate to a realistic frequency
    # Target: 50+ trades
    SIGNAL_THRESHOLD = 0.0003 # 0.03% expected return
    RISK_BUFFER = 0.0001     # 0.01% risk buffer
    
    print(f"Account Size: ${ACCOUNT_SIZE:,.0f}")
    print(f"Position Size: ${LOT_SIZE_USD:,.0f} per trade")
    print(f"Signal Threshold: {SIGNAL_THRESHOLD*100:.3f}% expected return")

    # 2. Load Data
    data_path = Path('data/processed/XAU_15m.csv.gz')
    df = pd.read_csv(data_path, compression='gzip')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.sort_values('time').reset_index(drop=True)

    # 3. Select Segment (100,000 candles)
    segment_size = 100000
    segment_end = len(df) - 175200 - 1000 # End before training start
    segment_start = segment_end - segment_size
    test_df = df.iloc[segment_start:segment_end].copy().reset_index(drop=True)
    
    print(f"\nProcessing {len(test_df)} candles (approx. 4 years of data)...")

    # 4. Features & Predictions
    calc = SimpleFeatureCalculator()
    test_df = calc.calculate_all_features(test_df)
    
    model = QuantileRegressor()
    model.load_models('models/quantile_forecast')
    
    X = test_df[model.feature_names]
    preds = model.predict(X)
    test_df = pd.concat([test_df, preds], axis=1)

    # 5. Signals (Decision Engine with Expanded Logic)
    engine = DecisionEngine(median_threshold=SIGNAL_THRESHOLD, risk_buffer=RISK_BUFFER)
    test_df['signal'] = engine.generate_signal(test_df)
    
    # 6. PnL Calculation
    # Raw log return
    test_df['future_return'] = np.log(test_df['close']).shift(-HORIZON) - np.log(test_df['close'])
    
    # Dollar PnL = Position Size * Percentage Change
    # Note: For log returns, pct_change is approx np.exp(return) - 1
    test_df['pct_change'] = np.exp(test_df['future_return']) - 1
    test_df['dollar_pnl'] = test_df['signal'] * LOT_SIZE_USD * test_df['pct_change']
    
    # Filter for trades
    trades = test_df[test_df['signal'] != 0].copy()
    trades = trades.dropna(subset=['future_return'])

    # 7. Summary
    if len(trades) == 0:
        print("No trades found with current thresholds.")
        return

    win_rate = (trades['dollar_pnl'] > 0).mean()
    total_dollar_pnl = trades['dollar_pnl'].sum()
    avg_dollar_pnl = trades['dollar_pnl'].mean()
    
    print("\n" + "-" * 40)
    print("V2 RESULTS (USD)")
    print("-" * 40)
    print(f"Total Trades:      {len(trades)}")
    print(f"Win Rate:         {win_rate:.1%}")
    print(f"Total PnL (USD):  ${total_dollar_pnl:,.2f}")
    print(f"Avg PnL / Trade:  ${avg_dollar_pnl:,.2f}")
    print(f"Return on Account: {total_dollar_pnl/ACCOUNT_SIZE:.2%}")
    
    # Save trades
    Path('results').mkdir(exist_ok=True)
    trades[['time', 'close', 'q_0.1', 'q_0.5', 'q_0.9', 'signal', 'pct_change', 'dollar_pnl']].to_csv('results/backtest_v2_dollar.csv', index=False)
    print(f"\nDetailed trades saved to results/backtest_v2_dollar.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()
