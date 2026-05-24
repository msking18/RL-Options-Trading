import numpy as np
import pandas as pd
from typing import List, Dict, Any

def calculate_performance_metrics(equity_curve: List[float], risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculates key financial performance metrics from an equity curve.
    """
    # Standard keys to ensure consistency
    metrics = {
        "total_return_pct": 0.0,
        "ann_return_pct": 0.0,
        "ann_vol_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "calmar_ratio": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0
    }
    
    if not equity_curve or len(equity_curve) < 2:
        return metrics
        
    equity = np.array(equity_curve)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-6)
    
    if len(returns) == 0:
        return metrics

    total_return = (equity[-1] / equity[0]) - 1
    
    # Assume 5-min candles, 78 per day, 252 days per year
    freq = 252 * 78 
    
    # Corrected Annualized Return (Linear annualization for stability across different evaluation horizons)
    ann_return = total_return * (freq / len(equity))
    ann_vol = np.std(returns) * np.sqrt(freq)
    
    # Sharpe Ratio (with epsilon bound to prevent divide-by-zero or extreme scaling spikes)
    metrics["sharpe_ratio"] = (ann_return - risk_free_rate) / (ann_vol + 1e-4) if ann_vol > 0 else 0.0
    
    # Sortino Ratio (with epsilon bound)
    negative_returns = returns[returns < 0]
    downside_vol = np.std(negative_returns) * np.sqrt(freq) if len(negative_returns) > 0 else 0
    metrics["sortino_ratio"] = (ann_return - risk_free_rate) / (downside_vol + 1e-4) if downside_vol > 0 else 0.0
    
    # Max Drawdown
    cumulative_max = np.maximum.accumulate(equity)
    drawdowns = (cumulative_max - equity) / np.maximum(cumulative_max, 1e-6)
    max_drawdown = np.max(drawdowns)
    
    metrics["total_return_pct"] = total_return * 100
    metrics["ann_return_pct"] = ann_return * 100
    metrics["ann_vol_pct"] = ann_vol * 100
    metrics["max_drawdown_pct"] = max_drawdown * 100
    metrics["calmar_ratio"] = ann_return / max_drawdown if max_drawdown > 0 else 0
    metrics["win_rate_pct"] = np.mean(returns > 0) * 100
    
    # Profit Factor
    gross_profits = np.sum(returns[returns > 0])
    gross_losses = np.abs(np.sum(returns[returns < 0]))
    metrics["profit_factor"] = gross_profits / gross_losses if gross_losses > 0 else (float('inf') if gross_profits > 0 else 1.0)
    
    return metrics

class BuyAndHoldBenchmark:
    """Simulates a simple Buy & Hold strategy on the underlying index."""
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        
    def evaluate(self, prices: np.ndarray) -> List[float]:
        """Returns the equity curve for Buy & Hold."""
        if len(prices) == 0:
            return []
        
        # Calculate units bought
        units = self.initial_capital / prices[0]
        return (prices * units).tolist()

class StraddleSellerBenchmark:
    """
    Simulates a short straddle seller (Call + Put) strategy.
    Re-enters ATM straddle every day at open, closes at EOD.
    """
    def __init__(self, initial_capital: float, slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.slippage = slippage
        
    def evaluate(self, prices: np.ndarray) -> List[float]:
        """
        Simplified straddle seller logic.
        Ensures output length matches prices length.
        """
        if len(prices) == 0: return []
        
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        daily_start_price = prices[0]
        
        for i in range(1, len(prices)):
            # Every 78 steps, reset the daily_start_price
            if (i-1) % 78 == 0:
                daily_start_price = prices[i-1]
            
            p = prices[i]
            move_ratio = abs(p - daily_start_price) / daily_start_price
            # Assume daily straddle premium is ~1% of index price
            # captures ~0.0001 theta per 5-min step
            theta_gain = 0.0001 
            step_pnl = theta_gain - (move_ratio * 0.1) # Arbitrary sensitivity
            
            current_capital *= (1 + step_pnl - self.slippage/100)
            equity_curve.append(current_capital)
            
        return equity_curve

class MovingAverageCrossoverBenchmark:
    """EMA 20/50 crossover strategy on the underlying index."""
    def __init__(self, initial_capital: float, fast_window: int = 20, slow_window: int = 50):
        self.initial_capital = initial_capital
        self.fast_window = fast_window
        self.slow_window = slow_window
        
    def evaluate(self, prices: np.ndarray) -> List[float]:
        if len(prices) < self.slow_window:
            return [self.initial_capital] * len(prices)
            
        df = pd.DataFrame({'close': prices})
        df['ema_fast'] = df['close'].ewm(span=self.fast_window).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_window).mean()
        
        equity_curve = [self.initial_capital]
        current_capital = self.initial_capital
        position = 0 # 1 if long, 0 if flat
        
        for i in range(1, len(prices)):
            # Crossover logic
            if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and position == 0:
                position = 1 # Buy
            elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and position == 1:
                position = 0 # Sell
                
            if position == 1:
                # Approximate return
                ret = (prices[i] / prices[i-1]) - 1
                current_capital *= (1 + ret)
                
            equity_curve.append(current_capital)
            
        return equity_curve

class AlwaysATMStraddleBenchmark:
    """
    Simulates a strategy that is always long/short an ATM straddle.
    This one uses the actual pre-calculated option prices passed in.
    """
    def __init__(self, initial_capital: float, slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.slippage = slippage
        
    def evaluate(self, call_prices: np.ndarray, put_prices: np.ndarray) -> List[float]:
        if len(call_prices) == 0: return []
        
        equity_curve = [self.initial_capital]
        # Always long 1 lot of call and 1 lot of put
        # We re-evaluate the portfolio value every step
        
        # Initial units bought (total capital split between call and put)
        # For simplicity, we assume we buy as many straddles as initial capital allows
        combined_price = call_prices[0] + put_prices[0]
        units = (self.initial_capital * 0.8) / (combined_price * (1 + self.slippage)) # 20% margin
        cash = self.initial_capital - (units * combined_price * (1 + self.slippage))
        
        for i in range(1, len(call_prices)):
            current_val = units * (call_prices[i] + put_prices[i])
            equity_curve.append(cash + current_val)
            
        return equity_curve
