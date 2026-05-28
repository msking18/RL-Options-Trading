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
    DRAWDOWN_PENALTY_MULTIPLIER, MIN_TRADE_VALUE, STATIC_PER_SYMBOL_FEATURES,
    REWARD_SCALE, SIM_AGGRESSION, VOLATILITY_EXPANSION_BONUS,
    REGIME_VOL_LOW, REGIME_VOL_HIGH, MAX_STALE_DURATION, STALE_PENALTY_MULTIPLIER,
    ZERO_SHOT_SYMBOLS, EXIT_COOLDOWN_STEPS, REWARD_LOG_SCALE_MULTIPLIER
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
            'Max_Impact', 'Is_Event_Day', 'Vol_Percentile_252', 'Nifty_IT_Ret_Lag1'
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
                    # Added Put Greeks to tensor for symmetric observation and simulation
                    greek_cols.extend([
                        f'e{e}_s{i}_delta', f'e{e}_s{i}_gamma', f'e{e}_s{i}_theta', f'e{e}_s{i}_vega',
                        f'e{e}_s{i}_put_delta', f'e{e}_s{i}_put_theta'
                    ])
            tech_cols = ['RSI', 'ATR', 'Index_Vol', 'EMA_50', 'EMA_200']
            price_cols = ['e0_call_price', 'e0_put_price', 'e1_call_price', 'e1_put_price']
            custom_cols = ['volatility_skew', 'max_pain_distance']
            
            self.tensor_cols = ohlcv_cols + greek_cols + tech_cols + price_cols + custom_cols + self.external_features_cols
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
        
        self.idx_skew = self.col_to_idx['volatility_skew']
        self.idx_max_pain = self.col_to_idx['max_pain_distance']

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
        self.per_symbol_segment_size = (self.lookback_window * 8) + STATIC_PER_SYMBOL_FEATURES
        self.total_obs_size = (self.per_symbol_segment_size * self.total_slots) + 2 + len(self.external_features_cols) + 1 # +1 for Regime Signal
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
        
        # 3.5 Nifty IT Context (Sectoral proxy)
        try:
            it_df = self.db_manager.load_data("Nifty IT", start_date=self.start_date, end_date=self.end_date)
            if not it_df.empty:
                # Group by date to get daily returns for context
                it_df['Date_Only'] = pd.to_datetime(it_df['time']).dt.normalize()
                it_daily = it_df.groupby('Date_Only')['close'].last().reset_index()
                it_daily['Nifty_IT_Ret'] = np.log(it_daily['close'] / it_daily['close'].shift(1))
                it_daily['Nifty_IT_Ret_Lag1'] = it_daily['Nifty_IT_Ret'].shift(1)
                df = pd.merge(df, it_daily[['Date_Only', 'Nifty_IT_Ret_Lag1']], on='Date_Only', how='left')
        except Exception as e:
            print(f"Warning: Could not merge Nifty IT context: {e}")
        
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
                
                delta_call = norm.cdf(d1)
                df[f'e{e_idx}_s{i}_delta'] = delta_call
                df[f'e{e_idx}_s{i}_gamma'] = norm.pdf(d1) / (spot * vol * np.sqrt(T)) * 100
                theta_call = (- (spot * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) 
                             - rate * strike * np.exp(-rate * T) * norm.cdf(d2)) / 365.0
                df[f'e{e_idx}_s{i}_theta'] = theta_call
                df[f'e{e_idx}_s{i}_vega'] = (spot * norm.pdf(d1) * np.sqrt(T)) / 100.0
                
                # Added Put Specific Greeks for symmetry and accurate SL/TP simulation
                df[f'e{e_idx}_s{i}_put_delta'] = delta_call - 1.0
                df[f'e{e_idx}_s{i}_put_theta'] = (- (spot * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) 
                                                 + rate * strike * np.exp(-rate * T) * norm.cdf(-d2)) / 365.0
                
            # ATM Prices (e_idx case)
            d1_atm = (np.log(spot / atm_strike) + (rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2_atm = d1_atm - vol * np.sqrt(T)
            df[f'e{e_idx}_call_price'] = spot * norm.cdf(d1_atm) - atm_strike * np.exp(-rate * T) * norm.cdf(d2_atm)
            df[f'e{e_idx}_put_price'] = atm_strike * np.exp(-rate * T) * norm.cdf(-d2_atm) - spot * norm.cdf(-d1_atm)
            
            df[f'e{e_idx}_call_price'] = df[f'e{e_idx}_call_price'].clip(lower=0.01)
            df[f'e{e_idx}_put_price'] = df[f'e{e_idx}_put_price'].clip(lower=0.01)
            
        # 3. Custom Features (Volatility Skew and Max Pain Distance)
        # Skew proxy: rises during market downturns and when volatility is high
        ema_50 = df['EMA_50'].values if 'EMA_50' in df.columns else spot
        df['volatility_skew'] = 0.15 + 0.1 * (ema_50 - spot) / (spot * vol + 1e-6)
        
        # Max Pain Distance proxy: distance from spot to the nearest strike
        df['max_pain_distance'] = (spot - atm_strike) / spot
            
        return df

    def _get_obs(self):
        """
        Ultra-high-performance fully vectorized 3D observation generation.
        Expanded to include symmetric Greeks (Call/Put) and technical indicator history.
        """
        # 1. Global Portfolio State
        self.obs_buffer[0] = self.state_manager.total_capital / self.initial_capital
        self.obs_buffer[1] = self.state_manager.cash_balance / self.initial_capital
        
        # 2. Per-Slot Data (Vectorized 3D approach)
        active_slots = [i for i, active in enumerate(self.active_slots_mask) if active]
        
        data_section_end = self.total_obs_size - (len(self.external_features_cols) + 1)
        self.obs_buffer[2 : data_section_end] = 0.0
        
        if active_slots:
            active_sym_idxs = np.array([self.slot_to_idx[i] for i in active_slots])
            lb = self.lookback_window
            
            # 1. Batch Extract 3D [NumActive, Lookback, Features]
            windows = self.data_tensor[active_sym_idxs, self.current_step - lb : self.current_step, :]
            current_data = self.data_tensor[active_sym_idxs, self.current_step, :]
            
            # 2. Vectorized Normalization (3D Temporal History)
            # Channels: [Open, High, Low, Close, Volume, OI, RSI, Index_Vol]
            base_prices = windows[:, 0, self.idx_close].reshape(-1, 1, 1)
            base_prices[base_prices == 0] = 1.0
            
            # Combined 8-channel history: [OHLC(4), Vol(1), OI(1), RSI(1), IndexVol(1)] * lb
            # We reshape to [slots, channels * lb] for the buffer
            hist_8ch = np.zeros((len(active_slots), lb, 8), dtype=np.float32)
            hist_8ch[:, :, :4] = (windows[:, :, self.idx_open : self.idx_close+1] / base_prices)
            hist_8ch[:, :, 4] = np.log1p(windows[:, :, self.idx_volume]) / 15.0
            hist_8ch[:, :, 5] = np.log1p(windows[:, :, self.idx_oi]) / 20.0
            hist_8ch[:, :, 6] = (windows[:, :, self.idx_rsi] - 50.0) / 50.0
            hist_8ch[:, :, 7] = windows[:, :, self.idx_vol]
            hist_flat = hist_8ch.reshape(len(active_slots), -1)
            
            # 3. Symmetric Multi-Expiry Greeks (60 features total: 5 strikes x 2 expiries x 6 greeks)
            # Features: [Delta, Gamma, Theta, Vega, PutDelta, PutTheta]
            greeks_start = self.idx_timestamp + 1
            greeks = current_data[:, greeks_start : greeks_start + 60].copy()
            
            # Vectorized scaling for 6-feature segments
            # Indices relative to each strike block: [0:D, 1:G, 2:T, 3:V, 4:pD, 5:pT]
            for i in range(10): # 10 blocks (5 strikes x 2 expiries)
                base = i * 6
                greeks[:, base + 1] *= 0.1  # Gamma scaling
                greeks[:, base + 2] *= 0.01 # Theta scaling
                greeks[:, base + 3] *= 0.01 # Vega scaling
                greeks[:, base + 5] *= 0.01 # Put Theta scaling

            # 4. Expiry Option Prices (4 features: Call0, Put0, Call1, Put1)
            spots = current_data[:, self.idx_close]
            prices_idx = self.idx_call_e0
            opt_prices = current_data[:, prices_idx : prices_idx + 4] / spots.reshape(-1, 1)
            
            # 5. Technicals + Metrics (Static Snapshot)
            atrs = np.where(spots > 1.0, current_data[:, self.idx_atr] / spots, 0.0)
            
            # 6. Relative Returns (conviction signal)
            nifty_idx = self.symbol_to_idx.get("Nifty 50", 0)
            nifty_close = self.data_tensor[nifty_idx, self.current_step, self.idx_close]
            nifty_ret = (nifty_close / self.data_tensor[nifty_idx, self.current_step-1, self.idx_close]) - 1.0
            
            sym_rets = (spots / self.data_tensor[active_sym_idxs, self.current_step-1, self.idx_close]) - 1.0
            rel_rets = sym_rets - nifty_ret
            
            emas_trend = np.where(current_data[:, self.idx_ema50] > current_data[:, self.idx_ema200], 1.0, -1.0)
            
            # 7. Temporal Features
            dtes = current_data[:, self.idx_dte] / 7.0
            dows = current_data[:, self.idx_dow] / 6.0
            is_expiry = current_data[:, self.idx_is_expiry]
            
            # 8. Position Features
            entry_prices = self.state_manager.pos_entry_price[active_sym_idxs]
            pos_types = self.state_manager.pos_type[active_sym_idxs]
            durations = np.minimum(1.0, self.state_manager.pos_hold_dur[active_sym_idxs] / 100.0)
            
            # Construct Per-Symbol Feature Matrix for Fast Copying
            sym_features = np.zeros((len(active_slots), self.per_symbol_segment_size), dtype=np.float32)
            
            f_idx = 0
            # Temporal History (lb*8)
            sym_features[:, f_idx : f_idx + lb*8] = hist_flat; f_idx += lb*8
            # Greeks (60 Symmetric Greeks)
            sym_features[:, f_idx : f_idx + 60] = greeks; f_idx += 60
            # Opt Prices (4)
            sym_features[:, f_idx : f_idx + 4] = opt_prices; f_idx += 4
            # Pos Type (1), Dur (1)
            sym_features[:, f_idx] = (pos_types > 0).astype(np.float32); f_idx += 1
            sym_features[:, f_idx] = durations; f_idx += 1
            # Indicators (3) - RSI and Vol moved to history
            sym_features[:, f_idx] = atrs; f_idx += 1
            sym_features[:, f_idx] = rel_rets; f_idx += 1
            sym_features[:, f_idx] = emas_trend; f_idx += 1
            # Temporal (3)
            sym_features[:, f_idx] = dtes; f_idx += 1
            sym_features[:, f_idx] = dows; f_idx += 1
            sym_features[:, f_idx] = is_expiry; f_idx += 1
            
            # Risk Management Awareness (2)
            lot_sizes = np.array([self.lot_size_map.get(self.symbol_list[idx], 50) for idx in active_sym_idxs])
            sym_features[:, f_idx] = lot_sizes / 75.0; f_idx += 1
            sym_features[:, f_idx] = (spots * lot_sizes) / self.state_manager.total_capital; f_idx += 1
            
            # 9. Custom Options Features (2) - Volatility Skew and Max Pain Distance
            skews = current_data[:, self.idx_skew]
            max_pain_dists = current_data[:, self.idx_max_pain]
            sym_features[:, f_idx] = skews; f_idx += 1
            sym_features[:, f_idx] = max_pain_dists; f_idx += 1
            
            # Write to buffer by slot
            for j, slot_i in enumerate(active_slots):
                start, end = self.slot_slices[slot_i]
                self.obs_buffer[start : end] = sym_features[j]
                
        # 3. Global External Signals
        ext_len = len(self.external_features_cols)
        external_vals = self.data_tensor[0, self.current_step, self.idx_ext_start : self.idx_ext_start + ext_len].copy()
        rel_max_impact = self.idx_max_impact - self.idx_ext_start
        external_vals[rel_max_impact] /= 3.0
        self.obs_buffer[-(ext_len + 1):-1] = external_vals
        
        # Calculate Regime Signal based on Nifty 50 Volatility
        nifty_idx = self.symbol_to_idx.get("Nifty 50", 0)
        nifty_vol = self.data_tensor[nifty_idx, self.current_step, self.idx_vol]
        if nifty_vol < REGIME_VOL_LOW:
            regime_signal = 0.0
        elif nifty_vol <= REGIME_VOL_HIGH:
            regime_signal = 1.0
        else:
            regime_signal = 2.0
            
        self.obs_buffer[-1] = regime_signal
        
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
        not_on_cooldown = self.state_manager.exit_cooldown[active_sym_idxs] == 0
        
        # Check current volatility for asymmetric risk masking (Suggestion 3)
        curr_vols = self.data_tensor[active_sym_idxs, self.current_step, self.idx_vol]
        is_high_vol = curr_vols > VOL_SCALE_HIGH_THRESHOLD
        
        # Check current option prices for current week (e0)
        call_prices = self.data_tensor[active_sym_idxs, self.current_step, self.idx_call_e0]
        put_prices = self.data_tensor[active_sym_idxs, self.current_step, self.idx_call_e0 + 1]
        
        can_enter = flat & under_limit & not_on_cooldown
        m[active_indices[can_enter & (call_prices >= MIN_OPTION_PRICE)], 1] = True
        m[active_indices[can_enter & (put_prices >= MIN_OPTION_PRICE)], 2] = True
        
        # If high volatility, force Tight SL only (mask out all other risk options)
        # Risk Categories: 0:No SL, 1:Tight, 2:Conservative, 3:Standard
        high_vol_indices = active_indices[is_high_vol]
        if len(high_vol_indices) > 0:
            m[high_vol_indices, 4 + 0] = False # Mask out 'No SL'
            m[high_vol_indices, 4 + 2] = False # Mask out 'Conservative'
            m[high_vol_indices, 4 + 3] = False # Mask out 'Standard'
            
        # Suggestion 3: Disable risk_idx = 0 (No SL/TP) if volatility > VOL_SCALE_MED_THRESHOLD or in zero-shot regimes
        is_med_vol = curr_vols > VOL_SCALE_MED_THRESHOLD
        is_zero_shot = np.array([self.symbol_list[idx] in ZERO_SHOT_SYMBOLS for idx in active_sym_idxs])
        
        disable_no_sl = active_indices[is_med_vol | is_zero_shot]
        if len(disable_no_sl) > 0:
            m[disable_no_sl, 4 + 0] = False # Mask out 'No SL/TP'
        
        # Exit (3) is valid only if in position AND (hold duration >= MIN_HOLD_STEPS OR has a loss after >= 1 step)
        # Check for loss condition: current_option_price < entry_price
        in_pos = (pos_types > 0)
        has_loss = np.zeros_like(in_pos, dtype=bool)
        if np.any(in_pos):
            in_pos_sym_idxs = active_sym_idxs[in_pos]
            opened_e_idxs = self.state_manager.pos_expiry_index[in_pos_sym_idxs]
            p_types = self.state_manager.pos_type[in_pos_sym_idxs]
            
            # Map dynamic price columns
            opt_cols = np.where(opened_e_idxs == 0, self.idx_call_e0, self.idx_call_e1)
            opt_cols = np.where(p_types == 2, opt_cols + 1, opt_cols) # Put option price is the next column
            
            # Fetch current prices and entry prices
            curr_opt_prices = self.data_tensor[in_pos_sym_idxs, self.current_step, opt_cols]
            entry_prices = self.state_manager.pos_entry_price[in_pos_sym_idxs]
            
            has_loss[in_pos] = (curr_opt_prices < entry_prices)
            
        in_pos_ready = (pos_types > 0) & ((pos_durations >= MIN_HOLD_STEPS) | ((pos_durations >= 1) & has_loss))
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
        
        # Track portfolio high-water mark and current drawdown
        self.high_water_mark = max(self.high_water_mark, prev_capital)
        port_drawdown = (self.high_water_mark - prev_capital) / self.high_water_mark if self.high_water_mark > 0 else 0.0
        
        # 0. Daily Reset Check (Timestamp-based date change detection)
        if self.current_step > self.lookback_window:
            prev_ts = int(self.data_tensor[0, self.current_step - 1, self.idx_timestamp])
            curr_ts = int(self.data_tensor[0, self.current_step, self.idx_timestamp])
            
            # Use NumPy datetime64 for fast vectorized comparison
            prev_date = np.datetime64(prev_ts, 's').astype('datetime64[D]')
            curr_date = np.datetime64(curr_ts, 's').astype('datetime64[D]')
            
            if curr_date != prev_date:
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
                
                # Trade quality signal: count profitable exits for bonus applied later
                self._exit_quality_bonus = 0.0
                rel_exits = np.where(exits)[0]
                logs = self.state_manager.trade_logs[-len(exit_indices):]
                for j, log in enumerate(logs):
                    if log.get('pnl', 0) > 0:
                        bonus = 0.001
                        # Boost bonus if exiting profitably in high vol
                        if curr_data[rel_exits[j], self.idx_vol] > VOL_SCALE_MED_THRESHOLD:
                            bonus *= 2.0
                        self._exit_quality_bonus += bonus
            
            # Enters (Action 1: Buy Call, 2: Buy Put) - MUST respect masks
            buy_calls = (slot_actions[:, 0] == 1) & active_masks[:, 1]
            buy_puts = (slot_actions[:, 0] == 2) & active_masks[:, 2]
            
            num_entries = 0
            for i in np.where(buy_calls | buy_puts)[0]:
                sym = self.slot_to_symbol[active_indices[i]]
                e_idx = 0 # Forced restriction to Current Week
                risk_idx = slot_actions[i, 1] 
                sym_vol = curr_data[i, self.idx_vol]
                
                # Determine option column (Current Week)
                col = self.idx_call_e0 # e_idx 0
                if buy_puts[i]: col += 1
                
                # Dynamic ATR-scaled Stop-Loss and Take-Profit mapping
                if risk_idx == 0:
                    sl_pct, tp_pct = 0.0, 0.0
                else:
                    # Fetch ATR, Vol, and Close price for current symbol
                    spot = curr_data[i, self.idx_close]
                    atr = curr_data[i, self.idx_atr]
                    
                    # Call/Put delta sensitivity
                    delta_col = self.idx_greeks.get((0, 0, 'delta'), self.idx_timestamp + 1)
                    d_val = curr_data[i, delta_col]
                    sens = d_val if buy_calls[i] else (1.0 - d_val)
                    
                    # Approximate option price change for 1 ATR underlying movement
                    option_price = curr_data[i, col]
                    option_atr_move = max(1e-4, atr * sens)
                    atr_option_pct = option_atr_move / max(option_price, 1e-6)
                    
                    # Tighten stop-losses in high volatility or zero-shot regimes
                    is_volatile_or_zero_shot = (sym_vol > VOL_SCALE_MED_THRESHOLD) or (sym in ZERO_SHOT_SYMBOLS)
                    
                    if risk_idx == 1: # Tight
                        sl_pct = float(np.clip(1.5 * atr_option_pct, 0.10, 0.30 if is_volatile_or_zero_shot else 0.40))
                        tp_pct = float(np.clip(3.0 * atr_option_pct, 0.20, 0.60 if is_volatile_or_zero_shot else 0.80))
                    elif risk_idx == 2: # Conservative
                        sl_pct = float(np.clip(3.0 * atr_option_pct, 0.20, 0.45 if is_volatile_or_zero_shot else 0.60))
                        tp_pct = float(np.clip(6.0 * atr_option_pct, 0.40, 1.10 if is_volatile_or_zero_shot else 1.50))
                    else: # Standard (3)
                        sl_pct = float(np.clip(5.0 * atr_option_pct, 0.30, 0.60 if is_volatile_or_zero_shot else 0.80))
                        tp_pct = float(np.clip(10.0 * atr_option_pct, 0.60, 2.20 if is_volatile_or_zero_shot else 3.00))

                # Value-based position sizing to ensure commissions are diluted
                option_price = curr_data[i, col]
                base_lot_size = getattr(self, 'lot_size_map', {}).get(sym, 50)
                
                # Target units based on MIN_TRADE_VALUE
                target_units = MIN_TRADE_VALUE / max(option_price, 1e-6)
                # Round to nearest lot size
                lot_count = max(1, round(target_units / base_lot_size))
                
                # Dynamic Volatility scaling (Three-tier volatility scaling)
                if sym_vol > VOL_SCALE_HIGH_THRESHOLD:
                    vol_scale = 0.35   # Scale down to 35% size in high volatility (was 50%)
                elif sym_vol > VOL_SCALE_MED_THRESHOLD:
                    vol_scale = 0.65   # Scale down to 65% size in medium volatility (was 75%)
                else:
                    vol_scale = 1.0
                
                # Apply scaling factors to the calculated lot count (vol_scale only, dd_scale removed to prevent single-symbol specialization)
                final_lot_count = max(1, int(lot_count * vol_scale))
                lot_size = final_lot_count * base_lot_size
                
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
        
        # Tick down exit cooldown for all symbols
        self.state_manager.exit_cooldown = np.maximum(0, self.state_manager.exit_cooldown - 1)
        
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
            
            # Use symmetric delta logic: Put Delta = Call Delta - 1
            # For SL/TP approximation, we need the absolute sensitivity to the spot move.
            # Sensitivity for Calls is p_deltas, for Puts it is (1 - p_deltas)
            p_sensitivities = np.where(p_types == 1, p_deltas, 1.0 - p_deltas)
            
            # Relaxed Intra-Candle logic: Apply configurable SIM_AGGRESSION factor
            # to reduce 'noise' exits. (Reduced from 0.7 to 0.4 via ppo_config)
            opt_low_approx = close_prices - (spot_move_against * p_sensitivities * SIM_AGGRESSION)
            
            # Check for SL hits (Removed 1.05x aggressive buffer)
            entries = self.state_manager.pos_entry_price[a_idx]
            sl_prices = self.state_manager.pos_sl_price[a_idx]
            sl_hits = (sl_prices > 0) & (opt_low_approx <= sl_prices)
            
            # Check for TP hits (Removed 0.95x aggressive buffer)
            spot_move_favor = np.where(p_types == 1, spot_high - spot_close, spot_close - spot_low)
            spot_move_favor = np.maximum(0, spot_move_favor)
            opt_high_approx = close_prices + (spot_move_favor * p_sensitivities * SIM_AGGRESSION) 
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
                
                # Early SL Penalty: Discourage entries that hit SL within 3 steps of opening
                early_sl_hits = sl_hits[hitter_indices] & (self.state_manager.pos_hold_dur[exit_indices] <= 3)
                if np.any(early_sl_hits):
                    total_penalty += np.sum(early_sl_hits) * 0.005 
            
            # Remaining positions update curr_value
            remaining = ~ (sl_hits | tp_hits)
            if np.any(remaining):
                r_idx = a_idx[remaining]
                self.state_manager.pos_curr_value[r_idx] = close_prices[remaining] * self.state_manager.pos_qty[r_idx]
                
                # Update hold_dur and peaks
                pnl_pcts = (close_prices[remaining] - entries[remaining]) / entries[remaining]
                stale_durations = self.state_manager.pos_hold_dur[r_idx]
                self.state_manager.pos_peak_pnl[r_idx] = np.maximum(self.state_manager.pos_peak_pnl[r_idx], pnl_pcts)
                
                # Tiered Drawdown Penalty on remaining positions
                dd = self.state_manager.pos_peak_pnl[r_idx] - pnl_pcts
                # REVISED: Linear penalty to prevent exploding gradients. 
                # Normalize DD to a 0.0 - 1.0 scale over the threshold.
                hard_excess = np.maximum(0.0, dd - DRAWDOWN_THRESHOLD_HARD)
                soft_excess = np.maximum(0.0, dd - DRAWDOWN_THRESHOLD_SOFT)
                
                # Apply linear penalty, scaled by volatility if above threshold (Suggestion 3)
                pen = np.where(dd > DRAWDOWN_THRESHOLD_HARD, 
                               HARD_PENALTY_SCALE * hard_excess * DRAWDOWN_PENALTY_MULTIPLIER / 100.0,
                               np.where(dd > DRAWDOWN_THRESHOLD_SOFT, 
                                         SOFT_PENALTY_SCALE * soft_excess * DRAWDOWN_PENALTY_MULTIPLIER / 100.0, 
                                         0.0))
                
                # Dynamic Volatility scaling for drawdown penalties
                vols = nxt_data[remaining, self.idx_vol]
                vol_penalty_scale = np.where(vols > VOL_SCALE_HIGH_THRESHOLD, 2.0, 1.0)
                pen *= vol_penalty_scale
                
                # Volatility Expansion Bonus: Reward holding through high-volatility trends
                vol_bonus = np.where(vols > VOL_SCALE_MED_THRESHOLD, VOLATILITY_EXPANSION_BONUS, 0.0)
                
                
                # Stale Position Penalty (using updated constants)
                # Penalize if > MAX_STALE_DURATION and PnL is flat/negative
                stale_penalties = np.where(
                    (stale_durations > MAX_STALE_DURATION) & (pnl_pcts <= 0.005),
                    STALE_PENALTY_MULTIPLIER * (stale_durations - MAX_STALE_DURATION),
                    0.0
                )
                
                total_penalty += (np.sum(pen) + np.sum(stale_penalties) - np.sum(vol_bonus))

        self.state_manager._update_total_capital()
        new_capital = self.state_manager.total_capital
        
        # 4. Reward Logic
        raw_delta = (new_capital - prev_capital) / self.initial_capital
        # Symmetric log-scaling to damp down explosive options swings while preserving sign and direction
        scaled_delta = np.sign(raw_delta) * np.log1p(abs(raw_delta) * REWARD_SCALE * REWARD_LOG_SCALE_MULTIPLIER)
        reward = scaled_delta - total_penalty
        
        # Apply trade quality bonus accumulated from profitable exits
        if hasattr(self, '_exit_quality_bonus') and self._exit_quality_bonus > 0:
            reward += self._exit_quality_bonus
            self._exit_quality_bonus = 0.0
        
        # Apply entry penalties if any new positions were opened
        if 'num_entries' in locals() and num_entries > 0:
            reward -= num_entries * ENTRY_PENALTY
        
        # Patience bonus: reward flat holds (active slot, no position, chose Hold)
        if len(active_indices) > 0:
            flat_holds = np.sum(
                (slot_actions[:, 0] == 0) & (self.state_manager.pos_type[active_sym_idxs] == 0)
            )
            reward += flat_holds * PATIENCE_BONUS
        
        # Death Penalty — tightened to 50% loss for faster learning signal
        if new_capital < (self.initial_capital * 0.5):
            done = True
            reward -= 1.0 * REWARD_SCALE  # Scale-aware: -5.0 at REWARD_SCALE=5.0
            
        self._update_action_masks_vectorized()
        info = {"capital": new_capital}
        
        # 5. Reward Decomposition for diagnostics
        info["reward/capital_delta"] = float(scaled_delta)
        info["reward/penalty_total"] = float(total_penalty)
        info["reward/exit_quality"] = float(getattr(self, '_exit_quality_bonus', 0.0))
        info["reward/patience"] = float(flat_holds * PATIENCE_BONUS) if 'flat_holds' in locals() else 0.0
        info["reward/net"] = float(reward)
        
        if done:
            self.close_all_positions()
            info["trade_logs"] = self.state_manager.trade_logs
            if len(self.state_manager.trade_logs) > 0:
                wins = sum(1 for log in self.state_manager.trade_logs if log.get('pnl', 0) > 0)
                info["win_rate"] = wins / len(self.state_manager.trade_logs)
            else:
                info["win_rate"] = 0.0
            
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
