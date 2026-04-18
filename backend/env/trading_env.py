import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict

from backend.env.state_manager import TradingStateManager
from data.pipelines.db_manager import HistoricalDBManager
from data.preprocessing.greeks import calculate_black_scholes_greeks, calculate_black_scholes_price
from backend.db.local_db_manager import LocalDBManager
from backend.train.ppo_config import (
    INITIAL_CAPITAL, PATIENCE_BONUS, DRAWDOWN_THRESHOLD,
    DRAWDOWN_PENALTY_SCALE, VOL_SCALE_HIGH_THRESHOLD, VOL_SCALE_MED_THRESHOLD
)

class TradingEnv(gym.Env):
    """
    A reinforcement learning environment for options trading on NSE indices.
    Supports multi-symbol randomization and specific date ranges for evaluation.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self, 
        symbols: Union[str, List[str]] = "Nifty 50", 
        lookback_window: int = 30,
        initial_capital: float = INITIAL_CAPITAL,
        slippage: float = 0.001,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
        min_active_symbols: int = 5,
        max_active_symbols: int = 10,
        total_slots: int = 10,
        fixed_commission: float = 0.0,
        preloaded_tensor: Optional[np.ndarray] = None,
        col_to_idx: Optional[Dict[str, int]] = None
    ):
        super(TradingEnv, self).__init__()
        
        # Data and Parameters
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.slippage = slippage
        
        self.min_active_symbols = min_active_symbols
        self.max_active_symbols = max_active_symbols
        self.total_slots = total_slots
        self.fixed_commission = fixed_commission
        
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        self.db_manager = HistoricalDBManager()
        # Initialize state manager with ALL symbols (it handles positions internally)
        self.state_manager = TradingStateManager(self.symbols, initial_capital, slippage, fixed_commission)
        
        # Define fixed list of external features to ensure consistent observation space
        self.external_features_cols = [
            'ShanghaiComp_Ret_Lag1', 'Gold_Ret_Lag1', 'Silver_Ret_Lag1', 'DowJones_Ret_Lag1',
            'CAC40_Ret_Lag1', 'FTSE100_Ret_Lag1', 'DAX_Ret_Lag1', 'SP500_Ret_Lag1',
            'HangSeng_Ret_Lag1', 'Nikkei225_Ret_Lag1', 'Nasdaq100_Ret_Lag1',
            'Pos_Score_Lag1', 'Neg_Score_Lag1', 'Headline_Count_Lag1',
            'Max_Impact', 'Is_Event_Day'
        ]
        
        # 4. Finalize Data (Vectorized 3D Tensor for Performance)
        if preloaded_tensor is not None:
            print("Using pre-vectorized data tensor.")
            self.data_tensor = preloaded_tensor
            self.col_to_idx = col_to_idx
            # Required for StateManager
            self.symbol_list = self.symbols if isinstance(self.symbols, list) else [self.symbols]
            self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbol_list)}
            num_timesteps = self.data_tensor.shape[1]
        else:
            # Load Data for all symbols
            self.all_dfs = {}
            if preloaded_data is not None:
                print(f"Using preloaded data for {len(preloaded_data)} symbols.")
                self.all_dfs = preloaded_data
            else:
                print(f"Loading data from DB for symbols: {self.symbols}...")
                for sym in self.symbols:
                    df = self.db_manager.load_data(sym, start_date=self.start_date, end_date=self.end_date)
                    if df.empty:
                        print(f"Warning: No data found for {sym} in range {self.start_date} to {self.end_date}")
                        continue
                    df = self._merge_external_signals(df, sym)
                    df = self._precalculate_data(df, sym)
                    self.all_dfs[sym] = df

            print("Finalizing NumPy data tensors...")
            self.symbol_list = list(self.all_dfs.keys()) # Stable order
            self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbol_list)}
            
            # Determine all columns to be included in the tensor
            first_df = next(iter(self.all_dfs.values()))
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'oi']
            greek_cols = []
            for i in range(-2, 3):
                greek_cols.extend([f'strike_{i}_delta', f'strike_{i}_gamma', f'strike_{i}_theta', f'strike_{i}_vega'])
            tech_cols = ['RSI', 'ATR', 'Index_Vol', 'EMA_50', 'EMA_200']
            price_cols = ['atm_call_price', 'atm_put_price']
            
            self.tensor_cols = ohlcv_cols + greek_cols + tech_cols + price_cols + self.external_features_cols
            self.col_to_idx = {col: i for i, col in enumerate(self.tensor_cols)}
            
            num_symbols = len(self.symbol_list)
            num_timesteps = min(len(df) for df in self.all_dfs.values())
            num_features = len(self.tensor_cols)
            
            self.data_tensor = np.zeros((num_symbols, num_timesteps, num_features), dtype=np.float32)
            for i, sym in enumerate(self.symbol_list):
                df = self.all_dfs[sym].iloc[:num_timesteps]
                self.data_tensor[i] = df[self.tensor_cols].values
                
            print(f"Data tensor initialized: {self.data_tensor.shape}")
        
        # --- End NumPy Refactor ---

        # Action Space: MultiDiscrete(5, 5, ..., 5) for fixed total_slots
        # Each index: 0:Hold, 1:Buy Call, 2:Buy Put, 3:Exit Call, 4:Exit Put
        self.action_space = spaces.MultiDiscrete([5] * self.total_slots)
        
        self.current_step = self.lookback_window
        self.max_steps = num_timesteps - 1
        
        # Performance: Pre-cache Column Indices
        self.idx_open = self.col_to_idx['open']
        self.idx_high = self.col_to_idx['high']
        self.idx_low = self.col_to_idx['low']
        self.idx_close = self.col_to_idx['close']
        self.idx_volume = self.col_to_idx['volume']
        self.idx_oi = self.col_to_idx['oi']
        self.idx_rsi = self.col_to_idx['RSI']
        self.idx_atr = self.col_to_idx['ATR']
        self.idx_vol = self.col_to_idx['Index_Vol']
        self.idx_ema50 = self.col_to_idx['EMA_50']
        self.idx_ema200 = self.col_to_idx['EMA_200']
        self.idx_call = self.col_to_idx['atm_call_price']
        self.idx_put = self.col_to_idx['atm_put_price']
        self.idx_ext_start = self.col_to_idx[self.external_features_cols[0]]
        self.idx_max_impact = self.col_to_idx['Max_Impact']

        # Pre-allocate Observation Buffer (NumPy array)
        self.per_symbol_segment_size = (self.lookback_window * 6) + (5 * 4) + 3 + 5
        self.total_obs_size = (self.per_symbol_segment_size * self.total_slots) + 2 + len(self.external_features_cols)
        self.obs_buffer = np.zeros(self.total_obs_size, dtype=np.float32)
        
        # Slice mapping for each slot to avoid repeated math
        # Each slot has a fixed segment size in the observation buffer
        self.slot_slices = []
        for i in range(self.total_slots):
            start = 2 + (i * self.per_symbol_segment_size)
            self.slot_slices.append((start, start + self.per_symbol_segment_size))
            
        self.observation_space = spaces.Box(low=-1e6, high=1e6, shape=(self.total_obs_size,), dtype=np.float32)

        # Pre-allocate Action Mask (MultiDiscrete total_slots * 5)
        self.action_mask_buf = np.zeros(self.total_slots * 5, dtype=bool)
        
        self.reset()
        for i in range(self.total_slots):
            start = 2 + (i * self.per_symbol_segment_size)
            end = start + self.per_symbol_segment_size
            self.slot_slices.append((start, end))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_size,), dtype=np.float32
        )
        
        # Track symbol-to-slot mapping and active mask for randomization
        self.active_slots_mask = [False] * self.total_slots
        self.slot_to_symbol = {} # Mapping slot index to symbol name
        self.slot_to_idx = {}    # Mapping slot index to symbol index in data_tensor
        
        # Performance Tracking
        self.high_water_mark = initial_capital
        
        # LOT SIZE MAPPING for NSE Indices
        self.lot_size_map = {
            "Nifty 50": 50,
            "Nifty Bank": 15,
            "Nifty Fin Service": 40,
            "Nifty Midcap Select": 75,
            "Nifty Next 50": 25,
            "Nifty 100": 50,
            "Nifty 500": 25,
            "Nifty IT": 50,
            "Nifty Auto": 50,
            "Nifty Pharma": 50
        }

    def _merge_external_signals(self, df, symbol):
        """Merges historical sentiment, macro, and event data into df."""
        df['parsed_time'] = pd.to_datetime(df['time'])
        df['Date_Only'] = df['parsed_time'].dt.normalize()
        
        # 1. Macro Data -> Calculate log returns, lag by 1 day (from SQLite)
        try:
            db_manager = LocalDBManager()
            start_str = df['Date_Only'].min().strftime('%Y-%m-%d')
            end_str = df['Date_Only'].max().strftime('%Y-%m-%d')
            
            macro_df = db_manager.get_macro_range(start_str, end_str)
            if not macro_df.empty:
                # Map column names back to expected Cased versions if needed by existing logic
                col_mapping = {
                    'shanghaicomp': 'ShanghaiComp', 'gold': 'Gold', 'silver': 'Silver',
                    'dowjones': 'DowJones', 'cac40': 'CAC40', 'ftse100': 'FTSE100',
                    'dax': 'DAX', 'sp500': 'SP500', 'hangseng': 'HangSeng',
                    'nikkei225': 'Nikkei225', 'nasdaq100': 'Nasdaq100'
                }
                macro_df = macro_df.rename(columns=col_mapping)
                macro_df['Date_Only'] = pd.to_datetime(macro_df['date']).dt.normalize()
                
                for col in col_mapping.values():
                    if col in macro_df.columns:
                        macro_df[f'{col}_Ret'] = np.log(macro_df[col] / macro_df[col].shift(1))
                        macro_df[f'{col}_Ret_Lag1'] = macro_df[f'{col}_Ret'].shift(1)
                macro_cols = ['Date_Only'] + [c for c in macro_df.columns if '_Ret_Lag1' in c]
                macro_df = macro_df[macro_cols]
            else:
                macro_df = pd.DataFrame(columns=['Date_Only'])
        except Exception as e:
            print(f"Warning: Could not load macro data from SQLite: {e}")
            macro_df = pd.DataFrame(columns=['Date_Only'])
            
        # 2. Sentiment Data -> Lag by 1 day (from SQLite)
        try:
            # Reusing same date strings
            sent_df = db_manager.get_sentiment_range(start_str, end_str)
            if not sent_df.empty:
                sent_df['Date_Only'] = pd.to_datetime(sent_df['Date']).dt.normalize()
                for col in ['Pos_Score', 'Neg_Score', 'Headline_Count']:
                    if col in sent_df.columns:
                        sent_df[f'{col}_Lag1'] = sent_df[col].shift(1)
                
                # Apply log1p normalization to Headline_Count_Lag1
                if 'Headline_Count_Lag1' in sent_df.columns:
                    sent_df['Headline_Count_Lag1'] = np.log1p(sent_df['Headline_Count_Lag1'])
                    
                sent_cols = ['Date_Only'] + [c for c in sent_df.columns if '_Lag1' in c]
                sent_df = sent_df[sent_cols]
            else:
                sent_df = pd.DataFrame(columns=['Date_Only'])
        except Exception as e:
            print(f"Warning: Could not load sentiment data from SQLite: {e}")
            sent_df = pd.DataFrame(columns=['Date_Only'])
            
        # 3. Economic Events -> No lag needed as upcoming events are known (from SQLite)
        try:
            raw_ev_df = db_manager.get_events_range(start_str, end_str)
            db_manager.close() # Finished with DB
            
            if not raw_ev_df.empty:
                raw_ev_df['Date_Only'] = raw_ev_df['timestamp'].dt.normalize()
                
                # Simplified daily features
                ev_features = raw_ev_df.groupby('Date_Only').agg({
                    'impact_level': 'max'
                }).reset_index().rename(columns={'impact_level': 'Max_Impact'})
                ev_features['Is_Event_Day'] = 1
                ev_df = ev_features[['Date_Only', 'Max_Impact', 'Is_Event_Day']]
            else:
                ev_df = pd.DataFrame(columns=['Date_Only'])
        except Exception as e:
            print(f"Warning: Could not load event data from SQLite: {e}")
            ev_df = pd.DataFrame(columns=['Date_Only'])
            
        # Merge all into df
        df = pd.merge(df, macro_df, on='Date_Only', how='left')
        df = pd.merge(df, sent_df, on='Date_Only', how='left')
        df = pd.merge(df, ev_df, on='Date_Only', how='left')
        
        # 4. Technical Indicators (ATR, RSI, Rolling Vol)
        # ATR (14)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Rolling Log Volatility (20-period EMA of returns)
        df['Index_Ret'] = np.log(df['close'] / df['close'].shift(1))
        df['Index_Vol'] = df['Index_Ret'].ewm(span=20).std() * np.sqrt(252 * 78) # Annualized
        
        # EMA
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Ensure ALL deterministic external feature columns exist, fill with 0
        for col in self.external_features_cols:
            if col not in df.columns:
                df[col] = 0.0
        
        # FFill missing and handle remaining NaNs (for rolling windows)
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        return df

    def _precalculate_data(self, df, symbol):
        """Pre-calculates technical indicators and Greeks using vectorized logic to optimize runtime."""
        from scipy.stats import norm
        
        if "Bank" in symbol:
            strike_step = 100
        else:
            strike_step = 50
            
        print(f"  - Vectorizing Greeks for {symbol}...")
        
        # 1. Calculate ATM Strike for the entire series
        spot = df['close'].values
        vol = df['Index_Vol'].values if 'Index_Vol' in df.columns else np.full_like(spot, 0.18)
        vol = np.maximum(vol, 1e-6) # Prevent divide-by-zero warnings
        dte = 5.0 # Constant for now
        
        atm_strike = np.round(spot / strike_step) * strike_step
        
        # 2. Iterate through relative strikes (only 5 iterations instead of len(df))
        for i in range(-2, 3):
            strike = atm_strike + (i * strike_step)
            
            # Use direct numpy/scipy logic here to avoid overhead of round() and dict creation in a loop
            T = dte / 365.0
            rate = 0.07
            
            # d1, d2
            d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            
            # Delta, Gamma, Theta, Vega
            df[f'strike_{i}_delta'] = norm.cdf(d1)
            df[f'strike_{i}_gamma'] = norm.pdf(d1) / (spot * vol * np.sqrt(T)) * 100
            
            # Theta (Call)
            theta_call = (- (spot * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) 
                         - rate * strike * np.exp(-rate * T) * norm.cdf(d2)) / 365.0
            df[f'strike_{i}_theta'] = theta_call
            
            # Vega
            df[f'strike_{i}_vega'] = (spot * norm.pdf(d1) * np.sqrt(T)) / 100.0
        
        # 3. Calculate ATM Prices for entire series
        # Using atm_strike (i=0 case)
        d1_atm = (np.log(spot / atm_strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2_atm = d1_atm - vol * np.sqrt(T)
        
        df['atm_call_price'] = spot * norm.cdf(d1_atm) - atm_strike * np.exp(-rate * T) * norm.cdf(d2_atm)
        df['atm_put_price'] = atm_strike * np.exp(-rate * T) * norm.cdf(-d2_atm) - spot * norm.cdf(-d1_atm)
        
        # Ensure minimum prices
        df['atm_call_price'] = df['atm_call_price'].clip(lower=0.01)
        df['atm_put_price'] = df['atm_put_price'].clip(lower=0.01)
            
        return df

    def _get_obs(self):
        """
        Ultra-high-performance fully vectorized 3D observation generation.
        Reduces Python overhead by performing heavy lifting in NumPy 3D blocks.
        """
        # 1. Global Portfolio State
        self.obs_buffer[0] = self.state_manager.total_capital / self.initial_capital
        self.obs_buffer[1] = self.state_manager.cash_balance / self.initial_capital
        
        # 2. Per-Slot Data (Vectorized 3D approach)
        active_slots = [i for i, active in enumerate(self.active_slots_mask) if active]
        
        # Zero out only the part of the buffer for symbols to avoid stale data from previous symbols
        # (External features and portfolio state are always overwritten)
        data_section_end = self.total_obs_size - len(self.external_features_cols)
        self.obs_buffer[2 : data_section_end] = 0.0
        
        if active_slots:
            active_sym_idxs = np.array([self.slot_to_idx[i] for i in active_slots])
            lb = self.lookback_window
            
            # 1. Batch Extract 3D [NumActive, Lookback, Features]
            windows = self.data_tensor[active_sym_idxs, self.current_step - lb : self.current_step, :]
            current_data = self.data_tensor[active_sym_idxs, self.current_step, :]
            
            # 2. Vectorized Normalization (3D)
            base_prices = windows[:, 0, self.idx_close].reshape(-1, 1, 1)
            base_prices[base_prices == 0] = 1.0
            
            norm_ohlc = (windows[:, :, self.idx_open : self.idx_close+1] / base_prices).reshape(len(active_slots), -1)
            norm_vol = (np.log1p(windows[:, :, self.idx_volume]) / 15.0).reshape(len(active_slots), -1)
            norm_oi = (np.log1p(windows[:, :, self.idx_oi]) / 20.0).reshape(len(active_slots), -1)
            
            # 3. Greeks Scaling (Vectorized Batch)
            greeks = current_data[:, self.idx_oi + 1 : self.idx_oi + 21].copy()
            greeks[:, 1::4] *= 0.1  # Gamma
            greeks[:, 2::4] *= 0.01 # Theta
            greeks[:, 3::4] *= 0.01 # Vega
            
            # 4. Technicals + Metrics (Already 3D)
            spots = current_data[:, self.idx_close]
            rsis = (current_data[:, self.idx_rsi] - 50.0) / 50.0
            atrs = np.where(spots > 0, current_data[:, self.idx_atr] / spots, 0.0)
            vols = current_data[:, self.idx_vol]
            emas_trend = np.where(current_data[:, self.idx_ema50] > current_data[:, self.idx_ema200], 1.0, -1.0)
            
            # 5. Position Features from StateManager (Vectorized Arrays)
            entry_prices = self.state_manager.pos_entry_price[active_sym_idxs]
            pos_types = self.state_manager.pos_type[active_sym_idxs]
            curr_vals = self.state_manager.pos_curr_value[active_sym_idxs]
            durations = np.minimum(1.0, self.state_manager.pos_hold_dur[active_sym_idxs] / 100.0)
            peak_pnls = self.state_manager.pos_peak_pnl[active_sym_idxs]
            
            # Map pos_type string logic back to option prices
            call_prices = current_data[:, self.idx_call]
            put_prices = current_data[:, self.idx_put]
            curr_option_prices = np.where(pos_types == 1, call_prices, put_prices)
            pnl_pcts = np.divide(curr_option_prices - entry_prices, entry_prices, 
                                 out=np.zeros_like(entry_prices), where=entry_prices > 0)
            trailing_drawdowns = peak_pnls - pnl_pcts
            
            # Construct Per-Symbol Feature Matrix for Fast Copying
            # Each sym has: [OHLC(lb*4), Vol(lb), OI(lb), Greeks(20), Val, Type, Dur, RSI, ATR, Vol, Trend, DD]
            # [N_ACTIVE, FEATURES_PER_SYM]
            sym_features = np.zeros((len(active_slots), self.per_symbol_segment_size), dtype=np.float32)
            
            f_idx = 0
            sym_features[:, f_idx : f_idx + lb*4] = norm_ohlc; f_idx += lb*4
            sym_features[:, f_idx : f_idx + lb] = norm_vol; f_idx += lb
            sym_features[:, f_idx : f_idx + lb] = norm_oi; f_idx += lb
            sym_features[:, f_idx : f_idx + 20] = greeks; f_idx += 20
            
            sym_features[:, f_idx] = curr_vals / self.initial_capital; f_idx += 1
            sym_features[:, f_idx] = (pos_types > 0).astype(np.float32); f_idx += 1
            sym_features[:, f_idx] = durations; f_idx += 1
            
            sym_features[:, f_idx] = rsis; f_idx += 1
            sym_features[:, f_idx] = atrs; f_idx += 1
            sym_features[:, f_idx] = vols; f_idx += 1
            sym_features[:, f_idx] = emas_trend; f_idx += 1
            sym_features[:, f_idx] = trailing_drawdowns; f_idx += 1
            
            # Write to buffer by slot
            for j, slot_i in enumerate(active_slots):
                start, end = self.slot_slices[slot_i]
                self.obs_buffer[start : end] = sym_features[j]
                
        # 3. Global External Signals
        ext_len = len(self.external_features_cols)
        external_vals = self.data_tensor[0, self.current_step, self.idx_ext_start : self.idx_ext_start + ext_len].copy()
        rel_max_impact = self.idx_max_impact - self.idx_ext_start
        external_vals[rel_max_impact] /= 3.0
        self.obs_buffer[-ext_len:] = external_vals
        
        return self.obs_buffer # Removed .copy() for maximum speed

    def action_masks(self) -> np.ndarray:
        """
        Returns the pre-calculated vectorized action mask.
        Optimized by updating ONLY in step and reset.
        """
        return self.action_mask_buf

    def _update_action_masks_vectorized(self):
        """
        Fully vectorized action mask computation for all slots simultaneously.
        """
        # Reset entire mask (10, 5)
        m = self.action_mask_buf.reshape(self.total_slots, 5)
        m.fill(False)
        m[:, 0] = True # Hold is always valid if active
        
        active_indices = np.where(self.active_slots_mask)[0]
        if len(active_indices) == 0:
            return
            
        active_sym_idxs = np.array([self.slot_to_idx[i] for i in active_indices])
        pos_types = self.state_manager.pos_type[active_sym_idxs]
        
        # 0: Hold, 1: Buy Call, 2: Buy Put, 3: Exit Call, 4: Exit Put
        # Flat symbols (type 0)
        flat = (pos_types == 0)
        m[active_indices[flat], 1] = True
        m[active_indices[flat], 2] = True
        
        # Long Call (type 1)
        calls = (pos_types == 1)
        m[active_indices[calls], 3] = True
        
        # Long Put (type 2)
        puts = (pos_types == 2)
        m[active_indices[puts], 4] = True
        
        # Inactive slots already handled (only Hold is True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Randomly sample active symbols
        num_symbols_available = len(self.symbol_list)
        num_to_sample = self.np_random.integers(self.min_active_symbols, self.max_active_symbols + 1)
        num_to_sample = min(num_to_sample, num_symbols_available)
        
        sampled_indices = self.np_random.choice(range(num_symbols_available), size=num_to_sample, replace=False)
        
        self.active_slots_mask = [False] * self.total_slots
        self.slot_to_symbol = {}
        self.slot_to_idx = {}
        
        for i, idx in enumerate(sampled_indices):
            if i < self.total_slots:
                self.active_slots_mask[i] = True
                self.slot_to_idx[i] = idx
                self.slot_to_symbol[i] = self.symbol_list[idx]
        
        # 2. Reset state logic
        self.state_manager.reset()
        self.current_step = self.lookback_window
        self.high_water_mark = self.initial_capital
        
        # 3. Initialize Mask & Obs
        self._update_action_masks_vectorized()
        return self._get_obs(), {}

    def step(self, actions):
        """
        Ultra-high-performance vectorized step.
        Processes all slots simultaneously and updates state in bulk.
        """
        prev_capital = self.state_manager.total_capital
        self.state_manager.current_step = self.current_step
        
        # 1. Process Actions (Vectorized Logic with minimal loops)
        active_indices = np.where(self.active_slots_mask)[0]
        if len(active_indices) > 0:
            active_actions = actions[active_indices]
            active_sym_idxs = np.array([self.slot_to_idx[i] for i in active_indices])
            curr_data = self.data_tensor[active_sym_idxs, self.current_step]
            
            # Exits
            ext_calls = (active_actions == 3) & (self.state_manager.pos_type[active_sym_idxs] == 1)
            ext_puts = (active_actions == 4) & (self.state_manager.pos_type[active_sym_idxs] == 2)
            for i in np.where(ext_calls | ext_puts)[0]:
                self.state_manager.exit_position(self.slot_to_symbol[active_indices[i]], 
                                              curr_data[i, self.idx_call] if ext_calls[i] else curr_data[i, self.idx_put])
            
            # Enters
            buy_calls = (active_actions == 1) & (self.state_manager.pos_type[active_sym_idxs] == 0)
            buy_puts = (active_actions == 2) & (self.state_manager.pos_type[active_sym_idxs] == 0)
            for i in np.where(buy_calls | buy_puts)[0]:
                sym = self.slot_to_symbol[active_indices[i]]
                base_lot_size = getattr(self, 'lot_size_map', {}).get(sym, 50)
                
                # Volatility-conditional position sizing
                sym_vol = curr_data[i, self.idx_vol]  # Annualized vol
                if sym_vol > VOL_SCALE_HIGH_THRESHOLD:
                    vol_scale = 0.5    # High vol → half size
                elif sym_vol > VOL_SCALE_MED_THRESHOLD:
                    vol_scale = 0.75   # Medium vol → 3/4 size
                else:
                    vol_scale = 1.0    # Low vol → full size
                lot_size = max(1, int(base_lot_size * vol_scale))
                
                self.state_manager.enter_position(sym, 'LONG_CALL' if buy_calls[i] else 'LONG_PUT',
                                               curr_data[i, self.idx_call] if buy_calls[i] else curr_data[i, self.idx_put],
                                               curr_data[i, self.idx_close],
                                               quantity=lot_size)

        # 2. Transition
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        
        # 3. Vectorized Position Updates & Drawdown Penalty (NO per-step holding cost)
        act_mask = (self.state_manager.pos_type > 0)
        total_penalty = 0.0
        if np.any(act_mask):
            a_idx = np.where(act_mask)[0]
            nxt_data = self.data_tensor[a_idx, self.current_step]
            
            # Update curr_value
            p_types = self.state_manager.pos_type[a_idx]
            prices = np.where(p_types == 1, nxt_data[:, self.idx_call], nxt_data[:, self.idx_put])
            self.state_manager.pos_curr_value[a_idx] = prices * self.state_manager.pos_qty[a_idx]
            
            # Update hold_dur and peaks
            entries = self.state_manager.pos_entry_price[a_idx]
            pnl_pcts = np.divide(prices - entries, entries, 
                                 out=np.zeros_like(entries), where=entries > 0)
            self.state_manager.pos_peak_pnl[a_idx] = np.maximum(self.state_manager.pos_peak_pnl[a_idx], pnl_pcts)
            self.state_manager.pos_hold_dur[a_idx] += 1
            
            # Drawdown penalty ONLY (no per-step holding cost)
            pk = self.state_manager.pos_peak_pnl[a_idx]
            dd = pk - pnl_pcts
            total_penalty = np.sum(np.where(
                dd > DRAWDOWN_THRESHOLD, 
                DRAWDOWN_PENALTY_SCALE * ((dd - DRAWDOWN_THRESHOLD) * 100.0), 
                0.0
            ))

        self.state_manager._update_total_capital()
        new_capital = self.state_manager.total_capital
        
        # 4. Reward Logic
        reward = (new_capital - prev_capital) / self.initial_capital
        reward -= total_penalty
        
        # Patience bonus: reward flat holds (active slot, no position, chose Hold)
        if len(active_indices) > 0:
            flat_holds = np.sum(
                (active_actions == 0) & (self.state_manager.pos_type[active_sym_idxs] == 0)
            )
            reward += flat_holds * PATIENCE_BONUS
        
        # Death Penalty
        if new_capital < (self.initial_capital * 0.3):
            done = True
            reward -= 0.1
            
        self._update_action_masks_vectorized()
        info = {"capital": new_capital}
        
        if done:
            self.close_all_positions()
            info["trade_logs"] = self.state_manager.trade_logs
            
        return self._get_obs(), reward, done, False, info

    def close_all_positions(self):
        """Vectorized close-all for indices."""
        idx = min(self.current_step, self.max_steps)
        current_prices = {sym: self.data_tensor[i, idx, self.idx_call] for i, sym in enumerate(self.symbol_list)}
        self.state_manager.close_all_positions(current_prices)
