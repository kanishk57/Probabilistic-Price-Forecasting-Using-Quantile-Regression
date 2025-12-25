import pandas as pd
import numpy as np

class DecisionEngine:
    """
    Translates quantile forecasts into trading decisions.
    """
    def __init__(self, median_threshold=0.0005, risk_buffer=0.0002):
        self.median_threshold = median_threshold  # Minimum 0.05% expected move
        self.risk_buffer = risk_buffer            # Minimum buffer on the "wrong" side
        
    def generate_signal(self, predictions: pd.DataFrame) -> pd.Series:
        """
        Input: DataFrame with q_0.1, q_0.5, q_0.9
        Output: Series with 1 (Long), -1 (Short), or 0 (Hold)
        """
        signals = pd.Series(0, index=predictions.index)
        
        # Long Logic:
        # 1. Median forecast is bullish enough
        # 2. Lower bound (10th percentile) is not deeply negative (manageable risk)
        long_mask = (predictions['q_0.5'] > self.median_threshold) & \
                    (predictions['q_0.1'] > -self.risk_buffer)
        
        # Short Logic:
        # 1. Median forecast is bearish enough
        # 2. Upper bound (90th percentile) is not deeply positive
        short_mask = (predictions['q_0.5'] < -self.median_threshold) & \
                     (predictions['q_0.9'] < self.risk_buffer)
        
        signals[long_mask] = 1
        signals[short_mask] = -1
        
        return signals

    def evaluate_signals(self, signals, actual_returns):
        """
        Simple evaluation of signals against future actual returns.
        """
        pnl = signals * actual_returns
        trades = signals[signals != 0]
        
        if len(trades) == 0:
            return {{'trades': 0}}
            
        metrics = {{
            'total_trades': len(trades),
            'win_rate': (pnl[signals != 0] > 0).mean(),
            'total_pnl': pnl.sum(),
            'avg_pnl': pnl[signals != 0].mean()
        }}
        return metrics
