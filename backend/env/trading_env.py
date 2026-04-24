import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict
from datetime import datetime

from backend.env.state_manager import TradingStateManager
from data.pipelines.db_manager import HistoricalDBManager
from data.preprocessing.greeks import calculate_black_scholes_greeks, calculate_black_scholes_price
from backend.db.local_db_manager import LocalDBManager
from backend.train.ppo_config import (
    INITIAL_CAPITAL, PATIENCE_BONUS, DRAWDOWN_THRESHOLD_SOFT,
    DRAWDOWN_THRESHOLD_HARD, SOFT_PENALTY_SCALE, HARD_PENALTY_SCALE,
    VOL_SCALE_HIGH_THRESHOLD, VOL_SCALE_MED_THRESHOLD, SL_TP_CATEGORIES,
    MIN_HOLD_STEPS, MIN_OPTION_PRICE, ENTRY_PENALTY, MAX_TRADES_PER_DAY,
    DRAWDOWN_PENALTY_MULTIPLIER
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
        max_active_symbols: int = 6,
        total_slots: int = 6,
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
        # State manager will be initialized later once symbol_list is finalized
        
        # Define fixed list of external features to ensure consistent observation space
        self.external_features_cols = [
            'ShanghaiComp_Ret_Lag1', 'Gold_Ret_Lag1', 'Silver_Ret_Lag1', 'DowJones_Ret_Lag1',
            'CAC40_Ret_Lag1', 'FTSE100_Ret_Lag1', 'DAX_Ret_Lag1', 'SP500_Ret_Lag1',
            'HangSeng_Ret_Lag1', 'Nikkei225_Ret_Lag1', 'Nasdaq100_Ret_Lag1',
            'Pos_Score_Lag1', 'Neg_Score_Lag1', 'Headline_Count_Lag1',
            'Max_Impact', 'Is_Event_Day', 'Vol_Percentile_252'
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

            self.symbol_list = list(self.all_dfs.keys()) # Stable order
            
            # Validation: Ensure we have enough symbols
            num_loaded = len(self.symbol_list)
            if num_loaded < self.min_active_symbols:
                raise ValueError(f"Insufficient data: Only {num_loaded} symbols loaded, but MIN_SYMBOLS is {self.min_active_symbols}. "
                                 f"Check the 'No data found' warnings above for details.")
            
            if num_loaded < self.total_slots:
                print(f"Warning: Only {num_loaded} symbols loaded, which is fewer than {self.total_slots} slots. "
                      "Model efficiency may be reduced.")

            self.symbol_to_idx = {sym: i for i, sym in enumerate(self.symbol_list)}
            
            # Determine all columns to be included in the tensor
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume', 'oi', 'dte_0', 'day_of_week', 'is_expiry_day', 'timestamp']
            greek_cols = []
            for e in [0, 1]:
                for i in range(-2, 3):
                    greek_cols.extend([f'e{e}_s{i}_delta', f'e{e}_s{i}_gamma', f'e{e}_s{i}_theta', f'e{e}_s{i}_vega'])
            tech_cols = ['RSI', 'ATR', 'Index_Vol', 'EMA_50', 'EMA_200']
            price_cols = ['e0_call_price', 'e0_put_price', 'e1_call_price', 'e1_put_price']
            
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

        # Action Space: MultiDiscrete with 2 dimensions per slot:
        # 0: Action (0:Hold, 1:Buy Call, 2:Buy Put, 3:Exit)
        # 1: Risk (0:Tight, 1:Regular, 2:Aggressive, 3:None)
        self.action_space = spaces.MultiDiscrete([4, len(SL_TP_CATEGORIES)] * self.total_slots)
        
        self.current_step = self.lookback_window
        self.max_steps = num_timesteps - 1
        
        # Performance: Pre-cache Column Indices
        self.idx_open = self.col_to_idx['open']
        self.idx_high = self.col_to_idx['high']
        self.idx_low = self.col_to_idx['low']
        self.idx_close = self.col_to_idx['close']
        self.idx_volume = self.col_to_idx['volume']
        self.idx_oi = self.col_to_idx['oi']
        self.idx_dte = self.col_to_idx['dte_0']
        self.idx_dow = self.col_to_idx['day_of_week']
        self.idx_is_expiry = self.col_to_idx['is_expiry_day']
        self.idx_timestamp = self.col_to_idx['timestamp']

        self.idx_rsi = self.col_to_idx['RSI']
        self.idx_atr = self.col_to_idx['ATR']
        self.idx_vol = self.col_to_idx['Index_Vol']
        self.idx_ema50 = self.col_to_idx['EMA_50']
        self.idx_ema200 = self.col_to_idx['EMA_200']
        
        # Default to Current Day Expiry (e0) for quick access
        self.idx_call_e0 = self.col_to_idx['e0_call_price']
        self.idx_put_e0 = self.col_to_idx['e0_put_price']
        self.idx_call_e1 = self.col_to_idx['e1_call_price']
        self.idx_put_e1 = self.col_to_idx['e1_put_price']

        # Cache Greek indices dynamically to avoid hardcoded offsets
        self.idx_greeks = {}
        for e in [0, 1]:
            for s in range(-2, 3):
                for g in ['delta', 'gamma', 'theta', 'vega']:
                    col_name = f'e{e}_s{s}_{g}'
                    if col_name in self.col_to_idx:
                        self.idx_greeks[(e, s, g)] = self.col_to_idx[col_name]

        self.idx_ext_start = self.col_to_idx[self.external_features_cols[0]]
        self.idx_max_impact = self.col_to_idx['Max_Impact']
        self.idx_pos_score = self.col_to_idx['Pos_Score_Lag1']
        self.idx_neg_score = self.col_to_idx['Neg_Score_Lag1']

        # LOT SIZE MAPPING for NSE/BSE Indices
        self.lot_size_map = {
            "Nifty 50": 50, "Nifty Bank": 15, "Nifty Fin Service": 40,
            "Nifty Midcap Select": 75, "Nifty Next 50": 25, "SENSEX": 10, "SENSEX50": 15
        }
        self.slot_to_symbol = {} 
        self.slot_to_idx = {}    
        self.high_water_mark = initial_capital

        # Pre-allocate Observation Buffer (NumPy array)
        # Per symbol features adjusted: 
        # (lb*6) for OHLCV [lb*4] + Vol [lb] + OI [lb]
        # (2 * 5 * 4) for Greeks [2 expiries * 5 strikes * 4 greeks]
        # (2 * 2) for Call/Put Price [2 expiries * 2 types]
        # 3 for Pos Value, Type, Duration
        # 5 for RSI, ATR, Vol, Trend, DD
        # 3 for DTE, DayOfWeek, IsExpiry
        # 2 for Lot Size, Contract Value (Risk Management Features)
        self.per_symbol_segment_size = (self.lookback_window * 6) + (2 * 5 * 4) + (2 * 2) + 3 + 5 + 3 + 2
        self.total_obs_size = (self.per_symbol_segment_size * self.total_slots) + 2 + len(self.external_features_cols)
        self.obs_buffer = np.zeros(self.total_obs_size, dtype=np.float32)
        
        # Slice mapping for each slot to avoid repeated math
        # Each slot has a fixed segment size in the observation buffer
        self.slot_slices = []
        for i in range(self.total_slots):
            start = 2 + (i * self.per_symbol_segment_size)
            self.slot_slices.append((start, start + self.per_symbol_segment_size))
            
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_obs_size,), dtype=np.float32)

        # Pre-allocate Action Mask (MultiDiscrete total_slots * 8)
        # Dimensions: 4 (Action) + 4 (Risk) = 8 per slot
        self.action_mask_buf = np.zeros(self.total_slots * (4 + len(SL_TP_CATEGORIES)), dtype=bool)
        
        # Finalize State Manager with the actual loaded symbol list to ensure correct indexing
        self.state_manager = TradingStateManager(self.symbol_list, initial_capital, slippage, fixed_commission)
        
        self.reset()

        
        # Track symbol-to-slot mapping and active mask for randomization
        self.active_slots_mask = [False] * self.total_slots

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
        
        # Volatility Percentile Proxy (Min-Max scaled over last 252 days)
        # Using fast rolling min/max instead of expensive rolling rank
        vol_min = df['Index_Vol'].rolling(window=19656, min_periods=100).min()
        vol_max = df['Index_Vol'].rolling(window=19656, min_periods=100).max()
        df['Vol_Percentile_252'] = (df['Index_Vol'] - vol_min) / (vol_max - vol_min + 1e-6)
        
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

    def _get_expiry_dte(self, symbol, timestamp):
        """Calculates days to next expiry for a given symbol and timestamp."""
        # Key Expiry Day Mapping (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun)
        expiry_map = {
            "Nifty 50": 3,
            "Nifty Bank": 2,
            "Nifty Fin Service": 1,
            "Nifty Midcap Select": 0,
            "Nifty Next 50": 4,
            "SENSEX": 4,
            "SENSEX50": 4
        }
        target_day = expiry_map.get(symbol, 3) # Default to Thursday
        current_day = timestamp.dayofweek
        
        # Days until next target_day
        days_to_expiry = (target_day - current_day) % 7
        return float(days_to_expiry)

    def _precalculate_data(self, df, symbol):
        """Pre-calculates technical indicators and Greeks (Current and Next Week) using vectorized logic."""
        from scipy.stats import norm
        
        if "Bank" in symbol or "SENSEX" in symbol:
            strike_step = 100
        else:
            strike_step = 50
            
        print(f"  - Vectorizing Greeks & Multi-Expiry Data for {symbol}...")
        
        # 1. Base Logic
        spot = df['close'].values
        vol = df['Index_Vol'].values if 'Index_Vol' in df.columns else np.full_like(spot, 0.18)
        vol = np.maximum(vol, 1e-6)
        
        parsed_times = pd.to_datetime(df['time'])
        dtes_0 = np.array([self._get_expiry_dte(symbol, t) for t in parsed_times])
        dtes_1 = dtes_0 + 7.0 # Next week's expiry
        
        df['dte_0'] = dtes_0
        df['day_of_week'] = parsed_times.dt.dayofweek.values.astype(float)
        df['is_expiry_day'] = (dtes_0 == 0).astype(float)
        df['timestamp'] = parsed_times.values.astype('datetime64[s]').astype('int64') # Unix seconds
        
        atm_strike = np.round(spot / strike_step) * strike_step
        rate = 0.07

        # 2. Iterate through expiries (Current and Next)
        for e_idx, dtes in enumerate([dtes_0, dtes_1]):
            # Clip DTE to prevent divide by zero near close (min 5 mins)
            T = np.maximum(dtes, 5.0/1440.0) / 365.0
            
            for i in range(-2, 3):
                strike = atm_strike + (i * strike_step)
                d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)
                
                df[f'e{e_idx}_s{i}_delta'] = norm.cdf(d1)
                df[f'e{e_idx}_s{i}_gamma'] = norm.pdf(d1) / (spot * vol * np.sqrt(T)) * 100
                theta_call = (- (spot * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) 
                             - rate * strike * np.exp(-rate * T) * norm.cdf(d2)) / 365.0
                df[f'e{e_idx}_s{i}_theta'] = theta_call
                df[f'e{e_idx}_s{i}_vega'] = (spot * norm.pdf(d1) * np.sqrt(T)) / 100.0
                
            # ATM Prices (e_idx case)
            d1_atm = (np.log(spot / atm_strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2_atm = d1_atm - vol * np.sqrt(T)
            df[f'e{e_idx}_call_price'] = spot * norm.cdf(d1_atm) - atm_strike * np.exp(-rate * T) * norm.cdf(d2_atm)
            df[f'e{e_idx}_put_price'] = atm_strike * np.exp(-rate * T) * norm.cdf(-d2_atm) - spot * norm.cdf(-d1_atm)
            
            df[f'e{e_idx}_call_price'] = df[f'e{e_idx}_call_price'].clip(lower=0.01)
            df[f'e{e_idx}_put_price'] = df[f'e{e_idx}_put_price'].clip(lower=0.01)
            
        return df

    def _get_obs(self):
        """
        Ultra-high-performance fully vectorized 3D observation generation.
        Expanded to include multi-expiry Greeks, prices, and temporal features.
        """
        # 1. Global Portfolio State
        self.obs_buffer[0] = self.state_manager.total_capital / self.initial_capital
        self.obs_buffer[1] = self.state_manager.cash_balance / self.initial_capital
        
        # 2. Per-Slot Data (Vectorized 3D approach)
        active_slots = [i for i, active in enumerate(self.active_slots_mask) if active]
        
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
            
            # Optimized slicing and reshaping
            norm_ohlc = (windows[:, :, self.idx_open : self.idx_close+1] / base_prices).reshape(len(active_slots), -1)
            norm_vol = (np.log1p(windows[:, :, self.idx_volume]) / 15.0).reshape(len(active_slots), -1)
            norm_oi = (np.log1p(windows[:, :, self.idx_oi]) / 20.0).reshape(len(active_slots), -1)
            
            # 3. Multi-Expiry Greeks (e0 and e1 - 40 features total)
            # Greek columns follow timestamp in our tensor_cols definition
            greeks_start = self.idx_timestamp + 1
            greeks = current_data[:, greeks_start : greeks_start + 40].copy()
            greeks[:, 1::4] *= 0.1  # Gamma scaling
            greeks[:, 2::4] *= 0.01 # Theta scaling
            greeks[:, 3::4] *= 0.01 # Vega scaling

            # 4. Expiry Option Prices (4 features: Call0, Put0, Call1, Put1)
            spots = current_data[:, self.idx_close]
            prices_idx = self.idx_call_e0
            opt_prices = current_data[:, prices_idx : prices_idx + 4] / spots.reshape(-1, 1)
            
            # 5. Technicals + Metrics (Already 3D)
            rsis = (current_data[:, self.idx_rsi] - 50.0) / 50.0
            atrs = np.where(spots > 1.0, current_data[:, self.idx_atr] / spots, 0.0)
            vols = current_data[:, self.idx_vol]
            emas_trend = np.where(current_data[:, self.idx_ema50] > current_data[:, self.idx_ema200], 1.0, -1.0)
            
            # 6. Temporal Features
            dtes = current_data[:, self.idx_dte] / 7.0
            dows = current_data[:, self.idx_dow] / 6.0
            is_expiry = current_data[:, self.idx_is_expiry]
            
            # 7. Position Features
            entry_prices = self.state_manager.pos_entry_price[active_sym_idxs]
            pos_types = self.state_manager.pos_type[active_sym_idxs]
            curr_vals = self.state_manager.pos_curr_value[active_sym_idxs]
            durations = np.minimum(1.0, self.state_manager.pos_hold_dur[active_sym_idxs] / 100.0)
            peak_pnls = self.state_manager.pos_peak_pnl[active_sym_idxs]
            
            # PnL Calculation relative to Entry
            # We need to know which expiry was traded for better PnL tracking in obs, 
            # but StateManager.pos_curr_value already has the value.
            sym_total_cap = self.initial_capital
            pnl_pcts = np.zeros_like(entry_prices)
            valid_pos = (entry_prices > 0)
            if np.any(valid_pos):
                # StateManager tracks curr_value based on the specific contract traded
                # pnl_pct = (curr_val / qty - entry) / entry
                pnl_pcts[valid_pos] = (curr_vals[valid_pos] / self.state_manager.pos_qty[active_sym_idxs][valid_pos] - entry_prices[valid_pos]) / entry_prices[valid_pos]
            
            pnl_pcts = np.clip(pnl_pcts, -2.0, 2.0)
            trailing_drawdowns = np.clip(peak_pnls - pnl_pcts, 0.0, 2.0)
            
            # Construct Per-Symbol Feature Matrix for Fast Copying
            sym_features = np.zeros((len(active_slots), self.per_symbol_segment_size), dtype=np.float32)
            
            f_idx = 0
            # lb*6
            sym_features[:, f_idx : f_idx + lb*4] = norm_ohlc; f_idx += lb*4
            sym_features[:, f_idx : f_idx + lb] = norm_vol; f_idx += lb
            sym_features[:, f_idx : f_idx + lb] = norm_oi; f_idx += lb
            # Greeks (40)
            sym_features[:, f_idx : f_idx + 40] = greeks; f_idx += 40
            # Opt Prices (4)
            sym_features[:, f_idx : f_idx + 4] = opt_prices; f_idx += 4
            # Pos Val (1), Type (1), Dur (1)
            sym_features[:, f_idx] = curr_vals / self.initial_capital; f_idx += 1
            sym_features[:, f_idx] = (pos_types > 0).astype(np.float32); f_idx += 1
            sym_features[:, f_idx] = durations; f_idx += 1
            # Indicators (5)
            sym_features[:, f_idx] = rsis; f_idx += 1
            sym_features[:, f_idx] = atrs; f_idx += 1
            sym_features[:, f_idx] = vols; f_idx += 1
            sym_features[:, f_idx] = emas_trend; f_idx += 1
            sym_features[:, f_idx] = trailing_drawdowns; f_idx += 1
            # Temporal (3)
            sym_features[:, f_idx] = dtes; f_idx += 1
            sym_features[:, f_idx] = dows; f_idx += 1
            sym_features[:, f_idx] = is_expiry; f_idx += 1
            
            # Risk Management Awareness (2)
            lot_sizes = np.array([self.lot_size_map.get(self.symbol_list[idx], 50) for idx in active_sym_idxs])
            sym_features[:, f_idx] = lot_sizes / 75.0; f_idx += 1
            sym_features[:, f_idx] = (spots * lot_sizes) / self.state_manager.total_capital; f_idx += 1
            
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
        Fully vectorized action mask computation for 2D action space (Action, Risk).
        """
        # Dimensions per slot: 4 (Action) + 4 (Risk) = 8
        stride = 4 + len(SL_TP_CATEGORIES)
        m = self.action_mask_buf.reshape(self.total_slots, stride)
        m.fill(False)
        m[:, 0] = True # Hold is always valid
        
        # Risk choices are always valid if slot is active
        m[:, 4:] = True # Risk dim
        
        active_indices = np.where(self.active_slots_mask)[0]
        if len(active_indices) == 0:
            return
            
        active_sym_idxs = np.array([self.slot_to_idx[i] for i in active_indices])
        pos_types = self.state_manager.pos_type[active_sym_idxs]
        pos_durations = self.state_manager.pos_hold_dur[active_sym_idxs]
        
        # Action dim: 0:Hold, 1:Buy Call, 2:Buy Put, 3:Exit
        # Entry (1, 2) is valid only if:
        # 1. Slot is flat
        # 2. Option price >= MIN_OPTION_PRICE
        # 3. Daily trade limit for symbol not reached
        flat = (pos_types == 0)
        under_limit = self.state_manager.trades_today[active_sym_idxs] < MAX_TRADES_PER_DAY
        
        # Check current option prices for current week (e0)
        call_prices = self.data_tensor[active_sym_idxs, self.current_step, self.idx_call_e0]
        put_prices = self.data_tensor[active_sym_idxs, self.current_step, self.idx_call_e0 + 1]
        
        m[active_indices[flat & under_limit & (call_prices >= MIN_OPTION_PRICE)], 1] = True
        m[active_indices[flat & under_limit & (put_prices >= MIN_OPTION_PRICE)], 2] = True
        
        # Exit (3) is valid only if in position AND hold duration >= MIN_HOLD_STEPS
        in_pos_ready = (pos_types > 0) & (pos_durations >= MIN_HOLD_STEPS)
        m[active_indices[in_pos_ready], 3] = True

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
        """Vectorized step supporting 2D actions (Action, Risk) and hardcoded e0 expiry."""
        prev_capital = self.state_manager.total_capital
        self.state_manager.current_step = self.current_step
        
        # 0. Daily Reset Check (Timestamp-based date change detection)
        # We use the timestamp at the current step to determine if we've crossed into a new day.
        # This is more robust than DOW for intraday data where DOW only changes weekly.
        if self.current_step > self.lookback_window:
            prev_ts = int(self.data_tensor[0, self.current_step - 1, self.idx_timestamp])
            curr_ts = int(self.data_tensor[0, self.current_step, self.idx_timestamp])
            
            # Check if dates differ
            if datetime.fromtimestamp(curr_ts).date() != datetime.fromtimestamp(prev_ts).date():
                print(f"[DEBUG] Day Change detected at step {self.current_step}: {datetime.fromtimestamp(prev_ts).date()} -> {datetime.fromtimestamp(curr_ts).date()}")
                self.state_manager.reset_daily_stats()
        
        # Universal timestamp for trade logs
        current_timestamp = int(self.data_tensor[0, self.current_step, self.idx_timestamp])
        
        # Map flat actions back to per-slot [total_slots, 2]
        action_matrix = actions.reshape(self.total_slots, 2)
        
        # 1. Process Actions
        active_indices = np.where(self.active_slots_mask)[0]
        if len(active_indices) > 0:
            slot_actions = action_matrix[active_indices] # [N, 2]
            active_sym_idxs = np.array([self.slot_to_idx[i] for i in active_indices])
            curr_data = self.data_tensor[active_sym_idxs, self.current_step]
            
            # Fetch current masks for these slots
            m = self.action_masks().reshape(self.total_slots, -1)
            active_masks = m[active_indices]
            
            # Universal contextual info for trade logs
            first_idx = active_sym_idxs[0]
            extra_info = {
                'sentiment_pos': float(self.data_tensor[first_idx, self.current_step, self.idx_pos_score]),
                'sentiment_neg': float(self.data_tensor[first_idx, self.current_step, self.idx_neg_score]),
                'max_impact': int(self.data_tensor[first_idx, self.current_step, self.idx_max_impact])
            }
            
            exits = (slot_actions[:, 0] == 3) & active_masks[:, 3]
            
            if np.any(exits):
                exit_indices = active_sym_idxs[exits]
                # Determine price: ALWAYS Current Week (e0) due to restriction for new trades,
                # but handle correctly if existing position is e1
                opened_e_idxs = self.state_manager.pos_expiry_index[exit_indices]
                p_types = self.state_manager.pos_type[exit_indices]
                
                # Vectorized column selection for exit prices
                exit_cols = np.where(opened_e_idxs == 0, self.idx_call_e0, self.idx_call_e1)
                exit_cols = np.where(p_types == 2, exit_cols + 1, exit_cols) # Put if type is 2
                
                # Fetch prices from curr_data (which is already indexed by active_sym_idxs)
                # We need the relative index in curr_data
                rel_exits = np.where(exits)[0]
                exit_prices = curr_data[rel_exits, exit_cols]
                
                self.state_manager.bulk_exit(exit_indices, exit_prices, timestamp=current_timestamp, extra_info=extra_info)
            
            # Enters (Action 1: Buy Call, 2: Buy Put) - MUST respect masks
            buy_calls = (slot_actions[:, 0] == 1) & active_masks[:, 1]
            buy_puts = (slot_actions[:, 0] == 2) & active_masks[:, 2]
            
            num_entries = 0
            for i in np.where(buy_calls | buy_puts)[0]:
                sym = self.slot_to_symbol[active_indices[i]]
                e_idx = 0 # Forced restriction to Current Week
                risk_idx = slot_actions[i, 1] 
                sl_pct, tp_pct = SL_TP_CATEGORIES[risk_idx]
                
                base_lot_size = getattr(self, 'lot_size_map', {}).get(sym, 50)
                sym_vol = curr_data[i, self.idx_vol]
                vol_scale = 0.5 if sym_vol > VOL_SCALE_HIGH_THRESHOLD else (0.75 if sym_vol > VOL_SCALE_MED_THRESHOLD else 1.0)
                lot_size = max(1, int(base_lot_size * vol_scale))
                
                col = self.idx_call_e0 # e_idx 0
                if buy_puts[i]: col += 1
                
                success = self.state_manager.enter_position(sym, 'LONG_CALL' if buy_calls[i] else 'LONG_PUT',
                                               curr_data[i, col], curr_data[i, self.idx_close],
                                               quantity=lot_size, sl_pct=sl_pct, tp_pct=tp_pct, expiry_idx=e_idx,
                                               timestamp=current_timestamp)
                if success:
                    num_entries += 1
                else:
                    # This might happen if mask was stale or state_manager hard limit reached
                    # print(f"[DEBUG] Trade entry failed for {sym} at step {self.current_step}")
                    pass

        # 2. Transition
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        
        # 3. Vectorized Position Updates & Intra-Candle SL Simulation
        act_mask = (self.state_manager.pos_type > 0)
        total_penalty = 0.0
        if np.any(act_mask):
            a_idx = np.where(act_mask)[0]
            # Increment hold duration for all active positions that survived the step transition
            self.state_manager.pos_hold_dur[a_idx] += 1
            
            nxt_data = self.data_tensor[a_idx, self.current_step]
            
            # Simulated Intra-Candle logic: Delta-based approximation of option High/Low
            spot_high = nxt_data[:, self.idx_high]
            spot_low = nxt_data[:, self.idx_low]
            spot_close = nxt_data[:, self.idx_close]
            spot_prev_close = self.data_tensor[a_idx, self.current_step - 1, self.idx_close]
            
            # Option price updates
            e_indices = self.state_manager.pos_expiry_index[a_idx]
            p_types = self.state_manager.pos_type[a_idx]
            
            # Next candle's OHLC prices for the options
            # Simplified: Option price moves by Index * Delta
            # We will use the e0/e1 call/put prices directly for the next candle's CLOSE,
            # but for High/Low we approximate using spot volatility.
            nxt_call_e0 = nxt_data[:, self.idx_call_e0]
            nxt_put_e0 = nxt_data[:, self.idx_put_e0]
            nxt_call_e1 = nxt_data[:, self.idx_call_e1]
            nxt_put_e1 = nxt_data[:, self.idx_put_e1]
            
            close_prices = np.zeros_like(p_types, dtype=np.float32)
            close_prices = np.where((e_indices == 0) & (p_types == 1), nxt_call_e0, close_prices)
            close_prices = np.where((e_indices == 0) & (p_types == 2), nxt_put_e0, close_prices)
            close_prices = np.where((e_indices == 1) & (p_types == 1), nxt_call_e1, close_prices)
            close_prices = np.where((e_indices == 1) & (p_types == 2), nxt_put_e1, close_prices)
            
            p_deltas = np.zeros_like(p_types, dtype=np.float32)
            for j, e_val in enumerate([0, 1]):
                mask_e = (e_indices == e_val)
                if np.any(mask_e):
                    # Use dynamic lookup for s0_delta
                    delta_col = self.idx_greeks.get((e_val, 0, 'delta'), self.idx_timestamp + 1 + (e_val * 20))
                    p_deltas[mask_e] = nxt_data[mask_e, delta_col]
            
            # Calls suffer when spot goes Low, Puts suffer when spot goes High
            # IMPORTANT: We measure distance from 'spot_close' (current candle's close)
            # because 'close_prices' (current option prices) already reflect the move from prev_close.
            # Using prev_close would double-count the delta move.
            spot_move_against = np.where(p_types == 1, spot_close - spot_low, spot_high - spot_close)
            spot_move_against = np.maximum(0, spot_move_against)
            
            # Relaxed Intra-Candle logic: Apply a 0.9x factor to price move against
            # to reduce 'noise' exits.
            SIM_AGGRESSION = 0.9 
            opt_low_approx = close_prices - (spot_move_against * p_deltas * SIM_AGGRESSION)
            
            # Check for SL hits (Removed 1.05x aggressive buffer)
            entries = self.state_manager.pos_entry_price[a_idx]
            sl_prices = self.state_manager.pos_sl_price[a_idx]
            sl_hits = (sl_prices > 0) & (opt_low_approx <= sl_prices)
            
            # Check for TP hits (Removed 0.95x aggressive buffer)
            spot_move_favor = np.where(p_types == 1, spot_high - spot_close, spot_close - spot_low)
            spot_move_favor = np.maximum(0, spot_move_favor)
            opt_high_approx = close_prices + (spot_move_favor * p_deltas * SIM_AGGRESSION) 
            tp_prices = self.state_manager.pos_tp_price[a_idx]
            tp_hits = (tp_prices > 0) & (opt_high_approx >= tp_prices)
            
            # Process Exits for SL/TP hits first
            hitter_indices = np.where(sl_hits | tp_hits)[0]
            if len(hitter_indices) > 0:
                exit_indices = a_idx[hitter_indices]
                exit_prices = np.where(sl_hits[hitter_indices], sl_prices[hitter_indices], tp_prices[hitter_indices])
                
                # Metadata for tracking
                hit_info = {
                    'sl_tp_hit': True,
                    'is_sl': sl_hits[hitter_indices].tolist(),
                    'is_tp': tp_hits[hitter_indices].tolist()
                }
                self.state_manager.bulk_exit(exit_indices, exit_prices, timestamp=current_timestamp, extra_info=hit_info)
            
            # Remaining positions update curr_value
            remaining = ~ (sl_hits | tp_hits)
            if np.any(remaining):
                r_idx = a_idx[remaining]
                self.state_manager.pos_curr_value[r_idx] = close_prices[remaining] * self.state_manager.pos_qty[r_idx]
                
                # Update hold_dur and peaks
                pnl_pcts = (close_prices[remaining] - entries[remaining]) / entries[remaining]
                self.state_manager.pos_peak_pnl[r_idx] = np.maximum(self.state_manager.pos_peak_pnl[r_idx], pnl_pcts)
                
                # Tiered Drawdown Penalty on remaining positions
                dd = self.state_manager.pos_peak_pnl[r_idx] - pnl_pcts
                pen = np.where(dd > DRAWDOWN_THRESHOLD_HARD, 
                               HARD_PENALTY_SCALE * ((dd - DRAWDOWN_THRESHOLD_HARD) * DRAWDOWN_PENALTY_MULTIPLIER),
                               np.where(dd > DRAWDOWN_THRESHOLD_SOFT, SOFT_PENALTY_SCALE * ((dd - DRAWDOWN_THRESHOLD_SOFT) * DRAWDOWN_PENALTY_MULTIPLIER), 0.0))
                total_penalty = np.sum(pen)

        self.state_manager._update_total_capital()
        new_capital = self.state_manager.total_capital
        
        # 4. Reward Logic
        reward = (new_capital - prev_capital) / self.initial_capital
        reward -= total_penalty
        
        # Apply entry penalties if any new positions were opened
        if 'num_entries' in locals() and num_entries > 0:
            reward -= num_entries * ENTRY_PENALTY
        
        # Patience bonus: reward flat holds (active slot, no position, chose Hold)
        if len(active_indices) > 0:
            flat_holds = np.sum(
                (slot_actions[:, 0] == 0) & (self.state_manager.pos_type[active_sym_idxs] == 0)
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
        
        current_prices = {}
        for i, sym in enumerate(self.symbol_list):
            p_type = self.state_manager.pos_type[i]
            if p_type == 0:
                current_prices[sym] = 0.0
                continue
                
            e_idx = self.state_manager.pos_expiry_index[i]
            col = self.idx_call_e0 if e_idx == 0 else self.idx_call_e1
            if p_type == 2: col += 1 # Put price is next col
            
            current_prices[sym] = float(self.data_tensor[i, idx, col])
            
        self.state_manager.close_all_positions(current_prices, timestamp=int(self.data_tensor[0, idx, self.idx_timestamp]))
