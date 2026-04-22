import numpy as np
from typing import Dict, Optional, List
from backend.train.ppo_config import COMMISSION_TIERS, COMMISSION_TIER_DEFAULT, MIN_OPTION_PRICE

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
        
        # SL/TP and Expiry Tracking
        self.pos_expiry_index = np.zeros(self.num_symbols, dtype=np.int8)
        self.pos_sl_price = np.zeros(self.num_symbols, dtype=np.float32)
        self.pos_tp_price = np.zeros(self.num_symbols, dtype=np.float32)
        
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
        self.pos_expiry_index.fill(0)
        self.pos_sl_price.fill(0.0)
        self.pos_tp_price.fill(0.0)

    def enter_position(self, symbol: str, pos_type: str, option_price: float, index_price: float, 
                      quantity: int = 1, sl_pct: float = 0.0, tp_pct: float = 0.0, expiry_idx: int = 0):
        idx = self.symbol_to_idx.get(symbol)
        if idx is None or self.pos_type[idx] != self.TYPE_NONE or option_price < MIN_OPTION_PRICE:
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
        self.pos_expiry_index[idx] = expiry_idx
        
        # Calculate SL/TP prices relative to execution_price
        if sl_pct > 0:
            self.pos_sl_price[idx] = execution_price * (1 - sl_pct)
        else:
            self.pos_sl_price[idx] = 0.0
            
        if tp_pct > 0:
            self.pos_tp_price[idx] = execution_price * (1 + tp_pct)
        else:
            self.pos_tp_price[idx] = 0.0
        
        self.cash_balance -= total_cost
        return True

    def _get_exit_commission(self, hold_duration: int) -> float:
        """Graduated commission: penalize short holds (scalps), reward patience."""
        for max_dur, multiplier in COMMISSION_TIERS:
            if hold_duration <= max_dur:
                return self.fixed_commission * multiplier
        return self.fixed_commission * COMMISSION_TIER_DEFAULT

    def get_bulk_exit_commissions(self, hold_durations: np.ndarray) -> np.ndarray:
        """Vectorized version of graduated commission logic."""
        commissions = np.full_like(hold_durations, self.fixed_commission * COMMISSION_TIER_DEFAULT, dtype=np.float32)
        # Apply tiers in reverse to ensure the lowest valid multiplier is applied correctly
        # or just iterate through tiers (there are only 3-4 tiers usually)
        for max_dur, multiplier in reversed(COMMISSION_TIERS):
            commissions[hold_durations <= max_dur] = self.fixed_commission * multiplier
        return commissions

    def bulk_exit(self, indices: np.ndarray, current_option_prices: np.ndarray, extra_info: dict = None):
        """
        Processes multiple liquidations in a single vectorized pass.
        Eliminates the Python for-loop overhead for SL/TP and Expiry hits.
        """
        if len(indices) == 0:
            return 0.0

        quantities = self.pos_qty[indices]
        execution_prices = current_option_prices * (1 - self.slippage_pct)
        hold_durs = self.pos_hold_dur[indices]
        
        exit_commissions = self.get_bulk_exit_commissions(hold_durs)
        entry_commissions = np.full_like(indices, self.fixed_commission, dtype=np.float32)
        
        exit_proceeds = (execution_prices * quantities) - exit_commissions
        entry_costs = (self.pos_entry_price[indices] * quantities) + entry_commissions
        realized_pnls = exit_proceeds - entry_costs
        
        self.cash_balance += np.sum(exit_proceeds)
        self.total_pnl += np.sum(realized_pnls)
        
        # Logging (We still need to append to trade_logs, which is a list. 
        # This is the only slow part left, but it's only called on exits)
        curr_step = getattr(self, 'current_step', 0)
        for i, idx in enumerate(indices):
            sym = self.symbols[idx]
            entry_val = float(self.pos_entry_price[idx] * quantities[i])
            log_entry = {
                'symbol': sym,
                'type': self.INV_TYPE_MAP[self.pos_type[idx]],
                'quantity': int(quantities[i]),
                'entry_time': int(self.pos_entry_time[idx]),
                'exit_time': int(curr_step),
                'hold_duration': int(hold_durs[i]),
                'entry_price': float(self.pos_entry_price[idx]),
                'exit_price': float(execution_prices[i]),
                'pnl': float(realized_pnls[i]),
                'pnl_pct': float(realized_pnls[i] / entry_val) if entry_val > 0 else 0.0,
                'commission_exit': float(exit_commissions[i])
            }
            if extra_info:
                log_entry.update(extra_info)
            self.trade_logs.append(log_entry)

        # Vectorized Reset
        self.pos_type[indices] = self.TYPE_NONE
        self.pos_qty[indices] = 0
        self.pos_entry_price[indices] = 0.0
        self.pos_hold_dur[indices] = 0
        self.pos_curr_value[indices] = 0.0
        self.pos_peak_pnl[indices] = 0.0
        self.pos_expiry_index[indices] = 0
        self.pos_sl_price[indices] = 0.0
        self.pos_tp_price[indices] = 0.0
        
        self._update_total_capital()
        return np.sum(realized_pnls)

    def exit_position(self, symbol: str, current_option_price: float, extra_info: dict = None):
        idx = self.symbol_to_idx.get(symbol)
        if idx is None or self.pos_type[idx] == self.TYPE_NONE:
            return 0.0
            
        quantity = self.pos_qty[idx]
        execution_price = current_option_price * (1 - self.slippage_pct)
        
        # Graduated commission: scalps pay more, patient holds pay less
        exit_commission = self._get_exit_commission(int(self.pos_hold_dur[idx]))
        entry_commission = self.fixed_commission  # Entry always pays base rate
        
        realized_pnl = (execution_price * quantity - exit_commission) - (self.pos_entry_price[idx] * quantity + entry_commission)
        
        self.cash_balance += (execution_price * quantity) - exit_commission
        self.total_pnl += realized_pnl
        
        # Log trade (maintained for post-run analysis)
        entry_val = float(self.pos_entry_price[idx] * quantity)
        log_entry = {
            'symbol': symbol,
            'type': self.INV_TYPE_MAP[self.pos_type[idx]],
            'quantity': int(quantity),
            'entry_time': int(self.pos_entry_time[idx]),
            'exit_time': getattr(self, 'current_step', 0),
            'hold_duration': int(self.pos_hold_dur[idx]),
            'entry_price': float(self.pos_entry_price[idx]),
            'exit_price': float(execution_price),
            'pnl': float(realized_pnl),
            'pnl_pct': float(realized_pnl / entry_val) if entry_val > 0 else 0.0,
            'commission_exit': float(exit_commission)
        }
        
        # Merge extra info (Context like sentiment/macro)
        if extra_info:
            log_entry.update(extra_info)
            
        self.trade_logs.append(log_entry)
        
        # Reset symbol state
        self.pos_type[idx] = self.TYPE_NONE
        self.pos_qty[idx] = 0
        self.pos_entry_price[idx] = 0.0
        self.pos_hold_dur[idx] = 0
        self.pos_curr_value[idx] = 0.0
        self.pos_peak_pnl[idx] = 0.0
        self.pos_expiry_index[idx] = 0
        self.pos_sl_price[idx] = 0.0
        self.pos_tp_price[idx] = 0.0
        
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
        Now uses bulk_exit for speed.
        """
        open_mask = (self.pos_type != self.TYPE_NONE)
        if not np.any(open_mask):
            return
            
        indices = np.where(open_mask)[0]
        prices = np.array([current_option_prices.get(self.symbols[idx], self.pos_curr_value[idx] / self.pos_qty[idx]) for idx in indices])
        
        self.bulk_exit(indices, prices)

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
