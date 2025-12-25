import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import joblib
from sklearn.model_selection import TimeSeriesSplit

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.simple_features import SimpleFeatureCalculator
from src.models.quantile_regressor import QuantileRegressor

def load_data(data_path=None):
    """Load XAU_15m data from processed files"""
    if data_path:
        p = Path(data_path)
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'time'})
            return df
            
    # Default locations
    defaults = [
        Path('data/processed/XAU_15m.parquet'),
        Path('data/processed/XAU_15m.csv.gz'),
        Path('data/XAU_15m.parquet'),
        Path('data/XAU_15m_data.csv')
    ]
    
    for p in defaults:
        if p.exists():
            print(f"Loading data from {p}")
            if p.suffix == '.parquet':
                df = pd.read_parquet(p)
            elif p.suffix == '.gz':
                df = pd.read_csv(p, compression='gzip')
            else:
                # Handle CSV with semicolon separator
                df = pd.read_csv(p, sep=';')
            
            # Normalize column names to lowercase
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Rename 'date' to 'time' if needed
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'time'})
                
            return df
            
    raise FileNotFoundError("Could not find XAU_15m data")

def prepare_targets(df, horizon=4):
    """
    Create target: Future Log Return over 'horizon' periods.
    """
    df = df.copy()
    # Log price
    log_close = np.log(df['close'])
    
    # Future return = log_close(t+h) - log_close(t)
    df['target_return'] = log_close.shift(-horizon) - log_close
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Train Probabilistic Price Forecasting Model')
    parser.add_argument('--horizon', type=int, default=4, help='Forecast horizon in candles (default: 4 = 1h for 15m)')
    parser.add_argument('--limit', type=int, default=50000, help='Limit number of rows for training to speed up dev')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Probabilistic Price Forecasting - Training Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        # Create dummy data if needed for testing, but better to fail
        return

    # Ensure time sorted
    if 'time' in df.columns:
        # Try to parse the time column - handle format like "2004.06.11 07:15"
        try:
            df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M')
        except:
            df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows.")
    
    # Limit size EARLY to speed up feature calculation
    if len(df) > args.limit:
        print(f"Limiting to last {args.limit} rows BEFORE feature calculation")
        df = df.iloc[-args.limit:].reset_index(drop=True)
    
    # 2. Calculate Features
    print("\nCalculating features...")
    calc = SimpleFeatureCalculator()
    # We can only calculate a subset or all. calculating all takes time but ensures we have everything.
    # To save time in this script, we could assume features are already there? 
    # But let's run it to be safe and consistent with the plan.
    df = calc.calculate_all_features(df)
    
    # 3. Prepare Target
    print(f"\nPreparing target for horizon {args.horizon}...")
    df = prepare_targets(df, horizon=args.horizon)
    
    # Drop NaNs created by lagging/differencing
    df = df.dropna().reset_index(drop=True)
    
    print(f"Final dataset shape: {df.shape}")
    
    # Select Features (numerical only for LightGBM usually, handle cats if needed)
    # Exclude time, OHLCV, and target columns
    exclude = ['time', 'open', 'high', 'low', 'close', 'volume', 'target_return']
               
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool]]
    
    print(f"Using {len(feature_cols)} features")
    
    X = df[feature_cols]
    y = df['target_return']
    
    # 4. Train with Walk-Forward Validation
    print("\nTraining Quantile Regression Models (Walk-Forward)...")
    
    # Using simple TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    quantiles = [0.1, 0.5, 0.9]
    model = QuantileRegressor(quantiles=quantiles)
    
    coverage_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_val)
        
        # Evaluate
        # Check if actual is between q_0.1 and q_0.9
        in_range = (y_val >= preds['q_0.1']) & (y_val <= preds['q_0.9'])
        coverage = in_range.mean()
        
        width = (preds['q_0.9'] - preds['q_0.1']).mean()
        mae_median = np.mean(np.abs(y_val - preds['q_0.5']))
        
        print(f"Fold {fold+1}: Coverage (10-90)={coverage:.1%}, Mean Width={width:.5f}, MAE Median={mae_median:.5f}")
        coverage_scores.append(coverage)
        
    print(f"\nAverage Coverage: {np.mean(coverage_scores):.1%}")
    
    # 5. Final Retrain on all data
    print("\nRetraining on full dataset...")
    model.fit(X, y)
    
    # Save
    Path('models').mkdir(exist_ok=True)
    model.save_models('models/quantile_forecast')
    print("Models saved to models/quantile_forecast_*.joblib")
    
    # 6. Example Prediction Output
    print("\nExample Forecasts (Last 5 candles):")
    last_X = X.iloc[-5:]
    last_preds = model.predict(last_X)
    last_actuals = y.iloc[-5:] # Note: these are actual future returns? No, y is aligned so y.iloc[-5] is log return of price at t vs t+horizon.
    # Actually, in real-time we don't know relevant y for the last 'horizon' candles yet if they are the very recent ones.
    # But here we are using historical data where we shifted.
    # Let's show the predictions.
    
    for i in range(5):
        idx = last_X.index[i]
        t = df.iloc[idx]['time']
        p = last_preds.iloc[i]
        # Convert log return back to percentage
        # exp(ret) - 1
        pct_low = (np.exp(p['q_0.1']) - 1) * 100
        pct_med = (np.exp(p['q_0.5']) - 1) * 100
        pct_high = (np.exp(p['q_0.9']) - 1) * 100
        
        print(f"{t}: Pred Return Range: [{pct_low:.2f}%, {pct_high:.2f}%] (Median: {pct_med:.2f}%)")

if __name__ == "__main__":
    main()
