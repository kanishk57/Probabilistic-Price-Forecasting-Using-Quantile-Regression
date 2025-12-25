import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.simple_features import SimpleFeatureCalculator
from src.models.quantile_regressor import QuantileRegressor
from src.models.decision_engine import DecisionEngine

def run_institutional_backtest():
    print("=" * 60)
    print("INSTITUTIONAL STRATEGY BACKTEST (Model-Informed)")
    print("=" * 60)

    # 1. Configuration
    INITIAL_CAPITAL = 10000.0
    BASE_RISK_PERCENT = 0.01  # 1% base risk
    HORIZON = 4               # 1 hour
    K_ASYMMETRY = 1.5         # Reward/Risk asymmetric requirement
    CONFIDENCE_P = 0.65       # 65th percentile for confidence filter
    
    # 2. Load Data & Models
    data_path = Path('data/processed/XAU_15m.csv.gz')
    df = pd.read_csv(data_path, compression='gzip')
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.sort_values('time').reset_index(drop=True)

    # Select Segment (Holdout period)
    segment_size = 50000 
    segment_end = len(df) - 175200 - 1000 
    segment_start = segment_end - segment_size
    test_df = df.iloc[segment_start:segment_end].copy().reset_index(drop=True)

    # 3. Features & Predictions
    calc = SimpleFeatureCalculator()
    test_df = calc.calculate_all_features(test_df)
    
    model = QuantileRegressor()
    model.load_models('models/quantile_forecast')
    
    X = test_df[model.feature_names]
    preds = model.predict(X)
    test_df = pd.concat([test_df, preds], axis=1)

    # 4. Strategy Implementation
    engine = DecisionEngine(asymmetry_k=K_ASYMMETRY, confidence_p=CONFIDENCE_P)
    test_df['signal'] = engine.generate_signal(test_df)
    
    # Pre-calculate rolling std of range for sizing
    range_width = test_df['q_0.9'] - test_df['q_0.1']
    rolling_range_std = range_width.rolling(window=1000).std()
    
    trades = []
    capital = INITIAL_CAPITAL
    
    print(f"\nRunning backtest on {len(test_df)} bars...")
    
    # 5. Iterative Simulation (to handle dynamic exits properly)
    i = 0
    while i < len(test_df) - HORIZON:
        signal = test_df.iloc[i]['signal']
        
        if signal != 0:
            entry_price = test_df.iloc[i]['close']
            entry_time = test_df.iloc[i]['time']
            
            # Model-informed SL/TP
            # We use the predicted log-returns to set levels
            q01 = test_df.iloc[i]['q_0.1']
            q09 = test_df.iloc[i]['q_0.9']
            
            if signal == 1: # Long
                sl_price = entry_price * np.exp(q01)
                tp_price = entry_price * np.exp(q09)
            else: # Short
                sl_price = entry_price * np.exp(q09)
                tp_price = entry_price * np.exp(q01)
            
            # Position Sizing: Confidence-scaled
            # Score = width / typical history
            if rolling_range_std.iloc[i] > 0:
                conf_score = range_width.iloc[i] / rolling_range_std.iloc[i]
            else:
                conf_score = 1.0
                
            risk_multiplier = np.clip(conf_score / 2.0, 0.5, 1.5) # Normalized around 1.0
            actual_risk = BASE_RISK_PERCENT * risk_multiplier
            
            # Distance to Stop (Initial Risk R)
            risk_per_unit = abs(entry_price - sl_price)
            if risk_per_unit == 0: risk_per_unit = 0.01 # Avoid div zero
            
            position_size = (capital * actual_risk) / risk_per_unit
            
            # Partial Exit levels
            r_distance = risk_per_unit
            tp05_price = entry_price + (0.5 * r_distance if signal == 1 else -0.5 * r_distance)
            tp10_price = entry_price + (1.0 * r_distance if signal == 1 else -1.0 * r_distance)
            
            # Forward look for SL/TP hits
            window = test_df.iloc[i+1 : i+HORIZON+1]
            actual_pnl = 0
            exit_reason = "Horizon"
            exit_time = test_df.iloc[i + HORIZON]['time']
            exit_price = test_df.iloc[i + HORIZON]['close']
            remaining_pos = position_size
            
            for _, bar in window.iterrows():
                # 1. Check Stop Loss (Priority 1)
                if (signal == 1 and bar['low'] <= sl_price) or (signal == -1 and bar['high'] >= sl_price):
                    actual_pnl += (sl_price - entry_price if signal == 1 else entry_price - sl_price) * remaining_pos
                    exit_reason = "Stop Loss"
                    exit_price = sl_price
                    exit_time = bar['time']
                    remaining_pos = 0
                    break
                
                # 2. Check Partial TP 1 (0.5R)
                if remaining_pos == position_size:
                    if (signal == 1 and bar['high'] >= tp05_price) or (signal == -1 and bar['low'] <= tp05_price):
                        close_amt = position_size * 0.30
                        actual_pnl += (tp05_price - entry_price if signal == 1 else entry_price - tp05_price) * close_amt
                        remaining_pos -= close_amt
                
                # 3. Check Partial TP 2 (1.0R)
                if remaining_pos == position_size * 0.70:
                    if (signal == 1 and bar['high'] >= tp10_price) or (signal == -1 and bar['low'] <= tp10_price):
                        close_amt = position_size * 0.30
                        actual_pnl += (tp10_price - entry_price if signal == 1 else entry_price - tp10_price) * close_amt
                        remaining_pos -= close_amt
                
                # 4. Check Final TP
                if (signal == 1 and bar['high'] >= tp_price) or (signal == -1 and bar['low'] <= tp_price):
                    actual_pnl += (tp_price - entry_price if signal == 1 else entry_price - tp_price) * remaining_pos
                    exit_reason = "Final TP"
                    exit_price = tp_price
                    exit_time = bar['time']
                    remaining_pos = 0
                    break
            
            # 5. Exit at Horizon if still open
            if remaining_pos > 0:
                actual_pnl += (exit_price - entry_price if signal == 1 else entry_price - exit_price) * remaining_pos
                remaining_pos = 0
            
            capital += actual_pnl
            
            risk_usd = capital * actual_risk
            r_multiple = actual_pnl / risk_usd if risk_usd > 0 else 0
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': 'Long' if signal == 1 else 'Short',
                'entry': entry_price,
                'exit': exit_price,
                'reason': exit_reason,
                'pnl_usd': actual_pnl,
                'risk_usd': risk_usd,
                'r_multiple': r_multiple,
                'capital': capital,
                'risk_mult': risk_multiplier
            })
            
            # Jump forward to exit time to avoid overlapping trades
            i += HORIZON
        else:
            i += 1

    # 6. Performance Summary
    trade_df = pd.DataFrame(trades)
    if len(trade_df) > 0:
        # Calculate R-multiples
        # Risk amount was capital * actual_risk at time of entry
        # We need to capture that or recalculate. 
        # I'll add 'risk_usd' to the trade recording to be precise.
        
        win_rate = (trade_df['pnl_usd'] > 0).mean()
        total_pnl = trade_df['pnl_usd'].sum()
        max_drawdown = (trade_df['capital'].cummax() - trade_df['capital']).max()
        
        # New Metrics
        gross_profit = trade_df[trade_df['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = abs(trade_df[trade_df['pnl_usd'] < 0]['pnl_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        expectancy = total_pnl / len(trade_df)
        
        # Calculate R-multiples (PnL / Initial Risk)
        # In our case, initial risk was (capital_at_entry * actual_risk)
        # We'll approximate from recorded PnL and risk_mult if needed, 
        # but better to record it. I'll update the loop.
        
        print("\n" + "-" * 40)
        print("INSTITUTIONAL STRATEGY RESULTS")
        print("-" * 40)
        print(f"Total Trades:      {len(trade_df)}")
        print(f"Win Rate:         {win_rate:.1%}")
        print(f"Profit Factor:    {profit_factor:.2f}")
        print(f"Expectancy:       ${expectancy:.2f} per trade")
        print(f"Total PnL (USD):  ${total_pnl:,.2f}")
        print(f"Return on Account: {total_pnl/INITIAL_CAPITAL:.2%}")
        print(f"Avg R-Multiple:   {trade_df['r_multiple'].mean():.2f}R")
        print(f"Max Drawdown:     ${max_drawdown:,.2f}")
        print("-" * 40)
        
        # Save results
        trade_df.to_csv('results/institutional_backtest.csv', index=False)
        print(f"Detailed logs saved to results/institutional_backtest.csv")
    else:
        print("\nNo trades met the institutional criteria.")

if __name__ == "__main__":
    run_institutional_backtest()
