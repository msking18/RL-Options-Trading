import numpy as np
import pandas as pd
from typing import Union

def calculate_log_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    Calculate log returns to ensure stationarity.
    Formula: ln(P_t / P_{t-1})
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    return np.log(prices[1:] / prices[:-1])

def calculate_ema_distance(current_price: float, history: Union[pd.Series, np.ndarray], span: int = 20) -> float:
    """
    Calculate percentage distance from the Exponential Moving Average.
    Used for detecting overbought/oversold conditions in a stationary way.
    """
    if isinstance(history, np.ndarray):
        history = pd.Series(history)
    
    ema = history.ewm(span=span, adjust=False).mean().iloc[-1]
    return ((current_price - ema) / ema) * 100.0

def normalize_volatility_skew(put_iv: float, call_iv: float) -> float:
    """
    Measures market fear/greed by calculating the difference in IV between Puts and Calls.
    """
    if call_iv == 0: return 0.0
    return (put_iv - call_iv) / call_iv

def scale_min_max(value: float, min_val: float, max_val: float) -> float:
    """
    Standard Min-Max scaling to 0-1 range.
    """
    if max_val == min_val: return 0.5
    return (value - min_val) / (max_val - min_val)

def robust_z_score(value: float, history: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Z-Score using Median and Median Absolute Deviation (MAD) for outlier resistance.
    """
    if isinstance(history, np.ndarray):
        history = pd.Series(history)
    
    median = history.median()
    mad = (history - median).abs().median()
    
    if mad == 0: return 0.0
    return 0.6745 * (value - median) / mad
