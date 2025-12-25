import pandas as pd
import numpy as np

class SimpleFeatureCalculator:
    """
    Simplified feature calculator for probabilistic price forecasting.
    Uses only statistical features: returns, volatility, z-scores, momentum.
    """
    
    def calculate_all_features(self, df):
        """
        Calculate statistical features for time-series forecasting.
        
        Args:
            df: OHLCV DataFrame with columns: time, open, high, low, close, volume
            
        Returns:
            DataFrame with added features
        """
        df = df.copy()
        
        print("Calculating price-based features...")
        df = self._calculate_price_features(df)
        
        print("Calculating return features...")
        df = self._calculate_returns_features(df)
        
        print("Calculating volatility features...")
        df = self._calculate_volatility_features(df)
        
        print("Calculating volume features...")
        df = self._calculate_volume_features(df)
        
        print("Calculating time features...")
        df = self._calculate_time_features(df)
        
        return df
    
    def _calculate_price_features(self, df):
        """Calculate basic price-based features"""
        df = df.copy()
        
        # Price range and body
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
        
        # Wicks
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_ratio'] = df['upper_wick'] / df['range'].replace(0, np.nan)
        df['lower_wick_ratio'] = df['lower_wick'] / df['range'].replace(0, np.nan)
        
        # Price position (where close is relative to high-low range)
        df['close_position'] = (df['close'] - df['low']) / df['range'].replace(0, np.nan)
        
        return df
    
    def _calculate_returns_features(self, df):
        """Calculate log returns and momentum features"""
        df = df.copy()
        
        # Log Returns: ln(Close_t / Close_{t-1})
        df['log_close'] = np.log(df['close'])
        df['log_return'] = df['log_close'].diff()
        
        # Lagged returns (past information)
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)
        
        # Rolling cumulative returns (momentum over different windows)
        for window in [5, 10, 20, 50]:
            df[f'return_sum_{window}'] = df['log_return'].rolling(window=window).sum()
            df[f'return_mean_{window}'] = df['log_return'].rolling(window=window).mean()
        
        # Return z-scores (standardized returns)
        for window in [20, 50, 100]:
            mean = df['log_return'].rolling(window=window).mean()
            std = df['log_return'].rolling(window=window).std()
            df[f'return_zscore_{window}'] = (df['log_return'] - mean) / std.replace(0, np.nan)
        
        # Cleanup intermediate
        df.drop(columns=['log_close'], inplace=True, errors='ignore')
        
        return df
    
    def _calculate_volatility_features(self, df):
        """Calculate volatility measures"""
        df = df.copy()
        
        # Simple volatility (std dev of returns)
        for window in [5, 10, 20, 50, 100]:
            df[f'volatility_{window}'] = df['log_return'].rolling(window=window).std()
        
        # Parkinson volatility (uses High/Low range)
        # More efficient estimator than close-to-close
        const = 1.0 / (4.0 * np.log(2.0))
        df['log_hl_sq'] = (np.log(df['high'] / df['low']))**2
        for window in [10, 20, 50]:
            df[f'parkinson_vol_{window}'] = np.sqrt(
                const * df['log_hl_sq'].rolling(window=window).mean()
            )
        
        # Volatility z-scores (is current volatility high or low?)
        for window in [20, 50]:
            vol = df[f'volatility_{window}']
            vol_mean = vol.rolling(window=window).mean()
            vol_std = vol.rolling(window=window).std()
            df[f'volatility_zscore_{window}'] = (vol - vol_mean) / vol_std.replace(0, np.nan)
        
        # Range z-score
        for window in [20, 50]:
            range_mean = df['range'].rolling(window=window).mean()
            range_std = df['range'].rolling(window=window).std()
            df[f'range_zscore_{window}'] = (df['range'] - range_mean) / range_std.replace(0, np.nan)
        
        # Cleanup
        df.drop(columns=['log_hl_sq'], inplace=True, errors='ignore')
        
        return df
    
    def _calculate_volume_features(self, df):
        """Calculate volume-based features if volume is available"""
        df = df.copy()
        
        if 'volume' not in df.columns:
            return df
        
        # Volume moving averages
        for window in [10, 20, 50]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # Volume z-scores
        for window in [20, 50]:
            vol_mean = df['volume'].rolling(window=window).mean()
            vol_std = df['volume'].rolling(window=window).std()
            df[f'volume_zscore_{window}'] = (df['volume'] - vol_mean) / vol_std.replace(0, np.nan)
        
        return df
    
    def _calculate_time_features(self, df):
        """Calculate time-based features"""
        df = df.copy()
        
        if 'time' not in df.columns:
            return df
        
        # Extract time components
        df['hour'] = pd.to_datetime(df['time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
        df['day_of_month'] = pd.to_datetime(df['time']).dt.day
        df['month'] = pd.to_datetime(df['time']).dt.month
        
        # Trading session indicators (for 24h markets like forex/crypto)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        return df
