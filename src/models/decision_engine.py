import pandas as pd
import numpy as np

class DecisionEngine:
    """
    Translates quantile forecasts into institutional-grade trading decisions.
    Focuses on uncertainty quantification and payoff asymmetry.
    """
    def __init__(self, median_threshold=0.0005, asymmetry_k=1.5, confidence_p=0.6):
        self.median_threshold = median_threshold  # Directional conviction
        self.asymmetry_k = asymmetry_k            # upside > k * downside
        self.confidence_p = confidence_p          # range_width percentile filter
        
    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Input: DataFrame with q_0.1, q_0.5, q_0.9
        Output: Series with 1 (Long), -1 (Short), or 0 (Hold)
        """
        # 1. Basic metrics
        range_width = df['q_0.9'] - df['q_0.1']
        upside = df['q_0.9']
        downside = df['q_0.1']
        
        # 2. Confidence Filter (Trade only when range is high compared to history)
        # Using a rolling window to determine the 'typical' range
        rolling_range_q = range_width.rolling(window=1000).quantile(self.confidence_p)
        
        signals = pd.Series(0, index=df.index)
        
        # LONG Entry Logic:
        # 1. Directional Conviction: Median forecast > threshold
        # 2. Asymmetric Payoff: Predicted upside > k * abs(predicted downside)
        # 3. Sufficient Confidence: Current forecast range > historical p-percentile
        long_mask = (df['q_0.5'] > self.median_threshold) & \
                    (upside > (self.asymmetry_k * abs(downside))) & \
                    (range_width > rolling_range_q)
        
        # SHORT Entry Logic:
        # 1. Directional Conviction: Median forecast < -threshold
        # 2. Asymmetric Payoff: Predicted downside < -k * predicted upside
        # 3. Sufficient Confidence: Current forecast range > historical p-percentile
        short_mask = (df['q_0.5'] < -self.median_threshold) & \
                     (abs(downside) > (self.asymmetry_k * upside)) & \
                     (range_width > rolling_range_q)
        
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
