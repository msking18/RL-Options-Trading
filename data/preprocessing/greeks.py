import numpy as np
from scipy.stats import norm
from typing import Dict, Union, Optional

def calculate_black_scholes_greeks(
    spot: float, 
    strike: float, 
    dte: float, 
    vol: float, 
    rate: float = 0.07, 
    option_type: str = 'CE'
) -> Dict[str, float]:
    """
    Calculate Option Greeks using the Black-Scholes-Merton model.
    
    Parameters:
    - spot: Current price of the underlying index.
    - strike: Strike price of the option.
    - dte: Days to Expiry (e.g., 5.0).
    - vol: Implied Volatility (annualized, e.g., 0.15 for 15%).
    - rate: Risk-free interest rate (annualized, defaults to 7% for India).
    - option_type: 'CE' for Call, 'PE' for Put.
    
    Returns:
    - A dictionary containing Delta, Gamma, Theta, Vega, and Rho.
    """
    # Normalize time to expiry (annualized)
    T = dte / 365.0
    
    # Handle edge case: exact expiry
    if T <= 0.00001 or vol <= 0.0001:
        return {
            "delta": 1.0 if (option_type == 'CE' and spot > strike) or (option_type == 'PE' and spot < strike) else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }

    # d1 and d2 calculations
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    # Delta
    if option_type == 'CE':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (Same for Call and Put)
    gamma = norm.pdf(d1) / (spot * vol * np.sqrt(T))

    # Vega (Same for Call and Put) - Reported as Change per 1% change in IV
    vega = (spot * norm.pdf(d1) * np.sqrt(T)) / 100.0

    # Theta - Reported as Change per 1 day decay
    if option_type == 'CE':
        theta = (- (spot * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) 
                 - rate * strike * np.exp(-rate * T) * norm.cdf(d2)) / 365.0
    else:
        theta = (- (spot * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) 
                 + rate * strike * np.exp(-rate * T) * norm.cdf(-d2)) / 365.0

    # Rho - Reported as Change per 1% change in interest rates
    if option_type == 'CE':
        rho = (strike * T * np.exp(-rate * T) * norm.cdf(d2)) / 100.0
    else:
        rho = (-strike * T * np.exp(-rate * T) * norm.cdf(-d2)) / 100.0

    return {
        "delta": round(float(delta), 4),
        "gamma": round(float(gamma), 6),
        "theta": round(float(theta), 4),
        "vega": round(float(vega), 4),
        "rho": round(float(rho), 4)
    }

def calculate_black_scholes_price(
    spot: float, 
    strike: float, 
    dte: float, 
    vol: float, 
    rate: float = 0.07, 
    option_type: str = 'CE'
) -> float:
    """Calculate theoretical Black-Scholes option price."""
    T = dte / 365.0
    if T <= 0:
        if option_type == 'CE':
            return max(0.0, spot - strike)
        else:
            return max(0.0, strike - spot)
            
    if vol <= 0.0001:
        # Intrinsic value at expiry
        if option_type == 'CE':
            return max(0.0, spot - strike) * np.exp(-rate * T)
        else:
            return max(0.0, strike - spot) * np.exp(-rate * T)

    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    if option_type == 'CE':
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * T) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        
    return max(0.01, round(float(price), 2))

def calculate_iv_rank(current_iv: float, iv_history: Union[list, np.ndarray]) -> float:
    """
    Calculate IV Rank: Where the current IV stands relative to the high/low of history.
    """
    low = np.min(iv_history)
    high = np.max(iv_history)
    if high == low: return 50.0
    return ((current_iv - low) / (high - low)) * 100.0

def calculate_iv_percentile(current_iv: float, iv_history: Union[list, np.ndarray]) -> float:
    """
    Calculate IV Percentile: The percentage of days in history where IV was lower than current.
    """
    count = np.sum(np.array(iv_history) < current_iv)
    return (count / len(iv_history)) * 100.0
