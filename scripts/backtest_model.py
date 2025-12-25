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

def main():
    print("=" * 60)
    print("Backtesting Probabilistic Forecasts")
    print("=" * 60)

    # 1. Load Data
    data_path = Path('data/processed/XAU_15m.csv.gz')
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, compression='gzip')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.sort_values('time').reset_index(drop=True)

    # 2. Select Out-of-Sample segment (before the training limit)
    # Training used last 175,200 rows. Total rows ~480k.
    # We'll take 20,000 rows ending 10,000 rows before the training start.
    train_limit = 175200
    train_start_idx = len(df) - train_limit
    
    test_size = 20000
    test_end_idx = train_start_idx - 1000 # Buffer
    test_start_idx = test_end_idx - test_size
    
    test_df = df.iloc[test_start_idx:test_end_idx].copy().reset_index(drop=True)
    print(f"Test period: {test_df['time'].min()} to {test_df['time'].max()} ({len(test_df)} candles)")

    # 3. Features
    print("Calculating features...")
    calc = SimpleFeatureCalculator()
    test_df = calc.calculate_all_features(test_df)
    
    # 4. Load Model and Predict
    model = QuantileRegressor()
    model.load_models('models/quantile_forecast')
    
    X = test_df[model.feature_names]
    print("Running predictions...")
    preds = model.predict(X)
    
    # Merge predictions
    test_df = pd.concat([test_df, preds], axis=1)
    
    # 5. Simple Decision Logic (Decision Layer preview)
    # Thresholds:
    # 1. Median Return (q_0.5) must be significant.
    # 2. Expected Risk (lower/upper bound) must be manageable.
    
    horizon = 4 # 1 hour
    threshold = 0.0005 # 0.05% expected return
    
    test_df['signal'] = 0
    # Long signal
    test_df.loc[(test_df['q_0.5'] > threshold) & (test_df['q_0.1'] > -threshold), 'signal'] = 1
    # Short signal
    test_df.loc[(test_df['q_0.5'] < -threshold) & (test_df['q_0.9'] < threshold), 'signal'] = -1
    
    # 6. PnL Calculation
    # Assume we hold for 'horizon' periods (4 candles)
    test_df['future_actual_return'] = np.log(test_df['close']).shift(-horizon) - np.log(test_df['close'])
    
    test_df['trade_pnl'] = test_df['signal'] * test_df['future_actual_return']
    
    # Only count rows where we had a signal and a future price
    trades = test_df[test_df['signal'] != 0].copy()
    trades = trades.dropna(subset=['future_actual_return'])
    
    # Metrics
    win_rate = (trades['trade_pnl'] > 0).mean()
    total_pnl = trades['trade_pnl'].sum()
    avg_pnl = trades['trade_pnl'].mean()
    sharpe = (trades['trade_pnl'].mean() / trades['trade_pnl'].std()) * np.sqrt(252 * 24) if len(trades) > 1 else 0
    
    print("\n" + "-" * 30)
    print("BACKTEST RESULTS")
    print("-" * 30)
    print(f"Total Trades:      {len(trades)}")
    print(f"Win Rate:         {win_rate:.1%}")
    print(f"Total Log PnL:    {total_pnl:.4f} (approx {np.exp(total_pnl)-1:.2%})")
    print(f"Avg PnL per Trade: {avg_pnl:.5f}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    
    # Save trades
    Path('results').mkdir(exist_ok=True)
    trades[['time', 'close', 'q_0.1', 'q_0.5', 'q_0.9', 'signal', 'future_actual_return', 'trade_pnl']].to_csv('results/backtest_trades.csv', index=False)
    print("\nTrades saved to results/backtest_trades.csv")

if __name__ == "__main__":
    main()
