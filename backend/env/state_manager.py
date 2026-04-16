import numpy as np
from typing import Dict, Optional, List

class TradingStateManager:
    """
    Manages the portfolio state, capital, and position logic using high-performance NumPy arrays.
    Eliminates Python dictionary overhead for RL training loops.
    """
    # Position Type Constants
    TYPE_NONE = 0
    TYPE_CALL = 1
    TYPE_PUT = 2
    
    TYPE_MAP = {None: 0, 'LONG_CALL': 1, 'LONG_PUT': 2}
    INV_TYPE_MAP = {0: None, 1: 'LONG_CALL', 2: 'LONG_PUT'}

    def __init__(self, symbols: List[str], initial_capital: float = 1000000.0, slippage: float = 0.001, fixed_commission: float = 0.0):
        self.symbols = symbols
        self.num_symbols = len(symbols)
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}
        
        self.initial_capital = initial_capital
        self.slippage_pct = slippage
        self.fixed_commission = fixed_commission
        
        # Initialize Array-based state
        self.pos_type = np.zeros(self.num_symbols, dtype=np.int8)
        self.pos_qty = np.zeros(self.num_symbols, dtype=np.int32)
        self.pos_entry_price = np.zeros(self.num_symbols, dtype=np.float32)
        self.pos_curr_value = np.zeros(self.num_symbols, dtype=np.float32)
        self.pos_hold_dur = np.zeros(self.num_symbols, dtype=np.int32)
        self.pos_peak_pnl = np.zeros(self.num_symbols, dtype=np.float32)
        self.pos_entry_time = np.zeros(self.num_symbols, dtype=np.int32)
        
        self.reset()

    def reset(self):
        self.cash_balance = self.initial_capital
        self.total_capital = self.initial_capital
        self.total_pnl = 0.0
        self.trade_logs = []
        
        self.pos_type.fill(self.TYPE_NONE)
        self.pos_qty.fill(0)
        self.pos_entry_price.fill(0.0)
        self.pos_curr_value.fill(0.0)
        self.pos_hold_dur.fill(0)
        self.pos_peak_pnl.fill(0.0)
        self.pos_entry_time.fill(0)

    def enter_position(self, symbol: str, pos_type: str, option_price: float, index_price: float, quantity: int = 1):
        idx = self.symbol_to_idx.get(symbol)
        if idx is None or self.pos_type[idx] != self.TYPE_NONE:
            return False
            
        execution_price = option_price * (1 + self.slippage_pct)
        total_cost = (execution_price * quantity) + self.fixed_commission
        
        if self.cash_balance < total_cost:
            return False
            
        self.pos_type[idx] = self.TYPE_MAP[pos_type]
        self.pos_qty[idx] = quantity
        self.pos_entry_price[idx] = execution_price
        self.pos_hold_dur[idx] = 0
        self.pos_curr_value[idx] = option_price * quantity
        self.pos_peak_pnl[idx] = 0.0
        self.pos_entry_time[idx] = getattr(self, 'current_step', 0)
        
        self.cash_balance -= total_cost
        return True

    def exit_position(self, symbol: str, current_option_price: float):
        idx = self.symbol_to_idx.get(symbol)
        if idx is None or self.pos_type[idx] == self.TYPE_NONE:
            return 0.0
            
        quantity = self.pos_qty[idx]
        execution_price = current_option_price * (1 - self.slippage_pct)
        
        realized_pnl = (execution_price * quantity - self.fixed_commission) - (self.pos_entry_price[idx] * quantity + self.fixed_commission)
        
        self.cash_balance += (execution_price * quantity) - self.fixed_commission
        self.total_pnl += realized_pnl
        
        # Log trade (maintained for post-run analysis)
        self.trade_logs.append({
            'symbol': symbol,
            'type': self.INV_TYPE_MAP[self.pos_type[idx]],
            'quantity': int(quantity),
            'entry_time': int(self.pos_entry_time[idx]),
            'exit_time': getattr(self, 'current_step', 0),
            'hold_duration': int(self.pos_hold_dur[idx]),
            'entry_price': float(self.pos_entry_price[idx]),
            'exit_price': float(execution_price),
            'pnl': float(realized_pnl)
        })
        
        # Reset symbol state
        self.pos_type[idx] = self.TYPE_NONE
        self.pos_qty[idx] = 0
        self.pos_entry_price[idx] = 0.0
        self.pos_hold_dur[idx] = 0
        self.pos_curr_value[idx] = 0.0
        self.pos_peak_pnl[idx] = 0.0
        
        self._update_total_capital()
        return realized_pnl

    def _update_total_capital(self):
        """Ultra-fast vectorized capital summation."""
        self.total_capital = self.cash_balance + np.sum(self.pos_curr_value)

    def get_state_vector(self) -> np.ndarray:
        """
        Returns a vectorized representation of the global and per-symbol state.
        Fully vectorized - no loops.
        """
        # Vectorized assembly of positional features
        # [value, is_active, hold_duration] for all symbols
        pos_values = self.pos_curr_value / self.initial_capital
        pos_active = (self.pos_type > 0).astype(np.float32)
        pos_durations = np.minimum(1.0, self.pos_hold_dur / 100.0)
        
        # Interleave features: [val0, act0, dur0, val1, act1, dur1, ...]
        symbol_features = np.stack([pos_values, pos_active, pos_durations], axis=1).flatten()
        
        global_state = np.array([
            self.total_capital / self.initial_capital,
            self.cash_balance / self.initial_capital
        ], dtype=np.float32)
        
        return np.concatenate([global_state, symbol_features])

    def close_all_positions(self, current_option_prices: Dict[str, float]):
        """
        Forcefully exit all open positions at the current provided prices.
        """
        for symbol in self.symbols:
            idx = self.symbol_to_idx[symbol]
            if self.pos_type[idx] != self.TYPE_NONE:
                price = current_option_prices.get(symbol, self.pos_curr_value[idx] / self.pos_qty[idx])
                self.exit_position(symbol, price)

    # Backward compatibility properties for TradingEnv's current _get_obs (temporary)
    @property
    def positions(self):
        """Allows TradingEnv to still access state via dict-like interface if needed."""
        return {
            s: {
                'type': self.INV_TYPE_MAP[self.pos_type[i]],
                'quantity': self.pos_qty[i],
                'entry_price': self.pos_entry_price[i],
                'current_value': self.pos_curr_value[i],
                'hold_duration': self.pos_hold_dur[i],
                'peak_pnl_pct': self.pos_peak_pnl[i]
            } for i, s in enumerate(self.symbols)
        }
