import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    size: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = 0.0
    outcome: str = None  # 'tp', 'sl', 'timeout'
    r_multiple: float = 0.0

class BacktestEngine:
    def __init__(self, 
                 initial_capital=10000,
                 risk_per_trade=0.01,
                 max_trades_per_day=3,
                 max_concurrent_trades=1):
        
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_day = max_trades_per_day
        self.max_concurrent = max_concurrent_trades
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.current_capital = initial_capital

    def run_backtest(self, df, predictions):
        """
        Run backtest using model predictions

        Args:
            df: DataFrame with OHLCV and features
            predictions: Dict with 'fill_prob', 'time_to_fill', 'trade_outcome_prob'
        """
        df = df.copy()
        df['fill_prob'] = predictions.get('fill_prob', 0)
        df['tp_prob'] = predictions.get('tp_prob', 0)

        active_trades = []
        trades_today = 0
        current_date = None

        for i in range(len(df)):
            row = df.iloc[i]

            # Reset daily trade count
            if current_date != pd.to_datetime(row['time']).date():
                current_date = pd.to_datetime(row['time']).date()
                trades_today = 0

            # Update active trades
            active_trades = self._update_active_trades(active_trades, row)

            # Check for new signals
            if (row.get('fvg_type') is not None and
                row['fill_prob'] > 0.65 and
                row['tp_prob'] > 0.60 and
                trades_today < self.max_trades_per_day and
                len(active_trades) < self.max_concurrent):
                
                # Enter trade
                trade = self._enter_trade(row)
                if trade:
                    active_trades.append(trade)
                    trades_today += 1

            # Track equity
            self.equity_curve.append({
                'time': row['time'],
                'equity': self.current_capital
            })

        # Close any remaining trades
        for trade in active_trades:
            self._close_trade(trade, {'time': df.iloc[-1]['time'], 'close': df.iloc[-1]['close']}, 'timeout')

        return self._calculate_metrics()

    def _enter_trade(self, row):
        """Enter a new trade based on FVG signal"""
        fvg_type = row['fvg_type']
        atr = row['atr']

        if fvg_type == 'bullish':
            entry = row['fvg_midpoint']
            stop_loss = entry - (1.5 * atr)
            take_profit = entry + (2.5 * atr)
            direction = 'long'
        else:
            entry = row['fvg_midpoint']
            stop_loss = entry + (1.5 * atr)
            take_profit = entry - (2.5 * atr)
            direction = 'short'

        # Position sizing
        risk_amount = self.current_capital * self.risk_per_trade
        pip_risk = abs(entry - stop_loss)
        position_size = risk_amount / pip_risk if pip_risk > 0 else 0

        if position_size <= 0:
            return None

        trade = Trade(
            entry_time=row['time'],
            entry_price=entry,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=position_size
        )

        return trade

    def _update_active_trades(self, active_trades, row):
        """Check if any active trades hit TP/SL"""
        remaining_trades = []

        for trade in active_trades:
            if trade.direction == 'long':
                # Check SL
                if row['low'] <= trade.stop_loss:
                    self._close_trade(trade, row, 'sl', trade.stop_loss)
                # Check TP
                elif row['high'] >= trade.take_profit:
                    self._close_trade(trade, row, 'tp', trade.take_profit)
                else:
                    remaining_trades.append(trade)

            else:  # short
                if row['high'] >= trade.stop_loss:
                    self._close_trade(trade, row, 'sl', trade.stop_loss)
                elif row['low'] <= trade.take_profit:
                    self._close_trade(trade, row, 'tp', trade.take_profit)
                else:
                    remaining_trades.append(trade)

        return remaining_trades

    def _close_trade(self, trade, row, outcome, exit_price=None):
        """Close a trade and update capital"""
        trade.exit_time = row['time']
        trade.exit_price = exit_price or row.get('close')
        trade.outcome = outcome

        if trade.direction == 'long':
            pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - trade.exit_price) * trade.size

        trade.pnl = pnl

        # Calculate R-multiple
        risk = abs(trade.entry_price - trade.stop_loss) * trade.size
        trade.r_multiple = pnl / risk if risk > 0 else 0

        self.current_capital += pnl
        self.trades.append(trade)

    def _calculate_metrics(self):
        """Calculate backtest performance metrics"""
        # Always return a tuple: (metrics_dict, trades_df, equity_df)
        # If no trades occurred, return empty DataFrames and zeroed metrics.
        trades_df = pd.DataFrame([vars(t) for t in self.trades]) if self.trades else pd.DataFrame()

        # Basic metrics defaults
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0]) if total_trades > 0 else 0
        losing_trades = len(trades_df[trades_df['pnl'] < 0]) if total_trades > 0 else 0

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if total_trades > 0 else 0
        total_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if total_trades > 0 else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        net_pnl = trades_df['pnl'].sum() if total_trades > 0 else 0
        total_return = (net_pnl / self.initial_capital) * 100 if self.initial_capital else 0

        # Equity curve analysis — use whatever equity points we collected (may be empty)
        equity_df = pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()
        if not equity_df.empty:
            equity_df['returns'] = equity_df['equity'].pct_change()

            sharpe_ratio = (
                equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
                if equity_df['returns'].std() > 0 else 0
            )

            # Drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (
                (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
            )
            max_drawdown = equity_df['drawdown'].min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        # Average R-multiple
        avg_r = trades_df['r_multiple'].mean() if total_trades > 0 else 0

        # Expectancy
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_pnl': net_pnl,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'avg_r_multiple': avg_r,
            'expectancy': expectancy,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

        return metrics, trades_df, equity_df
