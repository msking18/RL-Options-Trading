import os
import json

# --- Directory Configuration ---
MODEL_DIR = os.getenv("MODEL_DIR", "backend/models")
REPORT_DIR = os.getenv("REPORT_DIR", "reports")
EVAL_DIR = os.path.join(REPORT_DIR, "evaluations")
LOG_DIR = os.getenv("LOG_DIR", "ppo_trading_tensorboard")

# --- Training Configuration ---
TOTAL_TIMESTEPS = 5_000_000 # Increased budget for stabilization post-regression

# --- Annealing Configuration ---
INITIAL_LR = 2e-5               # Reduced from 3e-5 to lower clip fraction (was 0.35)
FINAL_LR = 1e-5                 # Terminal learning rate
ENT_COEF_MIN = 0.008            # Restored to May 18 working value for sufficient exploration

# --- Stability Guard ---
CLIP_FRACTION_THRESHOLD = 0.35  # Early-stop if clip_fraction exceeds this for CLIP_PATIENCE intervals
CLIP_PATIENCE = 10              # Consecutive violations before early stop (~160K steps at 2048 n_steps)
CHECKPOINT_FREQ = 500_000      # Save a checkpoint every N total timesteps
INITIAL_CAPITAL = 5_000_000.0
MIN_SYMBOLS = 1
MAX_SYMBOLS = 6
TOTAL_SLOTS = 6
LOOKBACK_WINDOW = 30 # Number of previous steps to include in observation
FIXED_COMMISSION = 20.0 # Per order side (Upstox)
BROKERAGE_PER_SIDE = 20.0 # Upstox flat brokerage fee per order side

# --- Realistic Indian Options Taxes & Frictions ---
STT_SELL_RATE = 0.000625              # 0.0625% on option premium (Sell side only)
NSE_TRANS_CHARGES_RATE = 0.00053      # 0.053% of option premium value (NSE)
GST_RATE = 0.18                       # 18% on (Brokerage + Exchange Charges)
SEBI_FEES_RATE = 0.000001             # 0.0001% of premium value (₹10/crore)
STAMP_DUTY_BUY_RATE = 0.00003         # 0.003% of premium value (Buy side only)

# --- Reward Tuning ---
PATIENCE_BONUS = 0.0001          # Small tiebreaker; ~4% of typical step reward at REWARD_SCALE=5.0
REWARD_SCALE = 5.0               # Restored: amplifies capital-delta for learnable gradient signal
REWARD_LOG_SCALE_MULTIPLIER = 10.0   # Scale multiplier inside log1p to squash high-variance outliers
VOLATILITY_EXPANSION_BONUS = 0.001  # Restored to May 18 value (proportional to REWARD_SCALE=5.0)
DRAWDOWN_THRESHOLD_SOFT = 0.10   # Relaxed from 0.07: 7% was too aggressive for 6-month OoS windows
DRAWDOWN_THRESHOLD_HARD = 0.20   # Relaxed from 0.15: 15% triggered on normal intraday fluctuations in long evals
SOFT_PENALTY_SCALE = 0.25        # Compromise between 0.2 (too weak) and 0.3 (crushed Standard OoS)
HARD_PENALTY_SCALE = 0.5         # Death Penalty Scale (Requested: 0.5)
MIN_OPTION_PRICE = 5.0          # Minimum price to allow trade entry
ENTRY_PENALTY = 0.0001          # Restored to May 18 value (proportional to REWARD_SCALE=5.0)
MIN_TRADE_VALUE = 250000.0       # Minimum trade value to dilute fixed commissions
DRAWDOWN_PENALTY_MULTIPLIER = 5.0  # Restored to May 18 value; /100.0 divisor re-added in trading_env.py

# --- Architecture Constants ---
# (2*5*4) Call Greeks + (2*2) Prices + 1 Pos + 6 Tech + 3 Temp + 2 Risk + 2 Custom = 76
STATIC_PER_SYMBOL_FEATURES = 76  # Includes Symmetric Put Greeks + Volatility Skew + Max Pain Distance
EXTERNAL_FEATURES_COUNT = 19    # 18 macro/sentiment/event features + 1 for Regime Signal

def get_obs_size(lookback, slots):
    """Calculates the total observation size for the given architecture."""
    per_symbol = (lookback * 8) + STATIC_PER_SYMBOL_FEATURES # Increased to 8 temporal channels
    return (per_symbol * slots) + 2 + EXTERNAL_FEATURES_COUNT

# Graduated commission tiers: {max_hold_duration: multiplier}
# Scalps (1 step) pay 10× base to force longer holding
COMMISSION_TIERS = [
    (1, 10.0),   # 1 step   → 10.0× commission (punish scalping to force longer signals)
    (2, 5.0),    # 2 steps  → 5.0× commission
    (4, 2.0),    # 3-4 steps → 2.0× commission
    (9, 1.0),    # 5-9 steps → 1.0× commission
]
COMMISSION_TIER_DEFAULT = 0.5    # 10+ steps → 0.5× commission

# --- Regime & Stale Penalties ---
MAX_STALE_DURATION = 200         # Generous window before penalizing stale positions
STALE_PENALTY_MULTIPLIER = 0.00005 # Very gentle — avoid exit panic
REGIME_VOL_LOW = 0.15
REGIME_VOL_HIGH = 0.25
EARLY_STOPPING_PATIENCE = 5

# --- Volatility Position Sizing ---
VOL_SCALE_HIGH_THRESHOLD = 0.35   # Lowered from 0.40 to trigger high-vol scaling earlier
VOL_SCALE_MED_THRESHOLD = 0.20    # Lowered from 0.25 to trigger med-vol scaling earlier

# --- Minimum Hold Period ---
MIN_HOLD_STEPS = 5              # Reverted from 7: MIN_HOLD=7 caused training overfitting (51% WR in-sample, 15% OoS)
EXIT_COOLDOWN_STEPS = 2         # Reverted from 3: longer cooldown narrowed strategy space causing OoS collapse
MAX_TRADES_PER_DAY = 5          # Hard limit on trade entries per symbol per day
SIM_AGGRESSION = 0.5            # Intra-candle SL/TP delta simulation factor

USE_CNN = True         # Toggle for 1D-CNN vs MlpPolicy

# --- Data Partitioning (2D Split) ---
TRAIN_START_DATE = "2024-04-01"
TRAIN_END_DATE   = "2025-09-30" # 18 months
VAL_START_DATE   = "2025-10-01" # Start of 6-month True OoS
VAL_END_DATE     = "2026-04-10"

# --- Symbol Configuration ---
# 5 Core Liquid Indices for Training
TRAINING_SYMBOLS = [
    "Nifty 50", "Nifty Bank", "Nifty Fin Service", "Nifty Midcap Select", "SENSEX", "SENSEX50"
]

# 1 symbols for Zero-Shot Robustness
ZERO_SHOT_SYMBOLS = [
    "Nifty Next 50"
]

# Evaluation-only (Hidden during training)
EVAL_ONLY_SYMBOLS = [
    "SENSEX", "Nifty Bank", "Nifty 50"
]

# SL/TP mapped as (StopLossPct, TakeProfitPct)
# 0: No SL/TP (Full discretionary)
# 1: Tight (20% SL, 40% TP)
# 2: Conservative (35% SL, 80% TP)
# 3: Standard (50% SL, 150% TP)
SL_TP_CATEGORIES = [
    (0.0, 0.0),          # No SL/TP (full discretion)
    (0.20, 0.40),        # Tight
    (0.35, 0.80),        # Conservative
    (0.50, 1.50)         # Standard
]
def load_best_params():
    """Loads tuned hyperparameters from JSON if available."""
    path = os.path.join(MODEL_DIR, "best_hyperparams.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load best params: {e}")
    return {}

def get_ppo_params():
    """
    Returns PPO hyperparameters for Stable-Baselines3.
    """
    params = {
        "learning_rate": INITIAL_LR, # Centralized — see INITIAL_LR constant
        "batch_size": 1024,
        "n_steps": 4096,
        "n_epochs": 10,             # Increased back to 10 to improve value network fitting (EV)
        "gamma": 0.98,              # Tuned to fit weekly options theta decay half-life (~3 hours)
        "gae_lambda": 0.95,
        "clip_range": 0.20,         # Widened from 0.15 to reduce clip fraction (was 0.35)
        "ent_coef": 0.01,            # Start higher for exploration, anneal to ENT_COEF_MIN=0.003
        "vf_coef": 1.0,             # Increased from default 0.5 to prioritize Value Function accuracy
        "verbose": 1,
        "device": "auto" # Use CUDA if available
    }
    
    # Overlay tuned params if available
    best_params = load_best_params()
    if best_params:
        print(f"Overlapping config with {len(best_params)} tuned hyperparameters.")
        params.update(best_params)
    
    if USE_CNN:
        from backend.train.feature_extractors import Trading1DCNN
        print("Using Architecture: 1D-CNN (Temporal Feature Extractor)")
        params["policy_kwargs"] = dict(
            features_extractor_class=Trading1DCNN,
            features_extractor_kwargs=dict(features_dim=512, lookback_window=LOOKBACK_WINDOW, total_slots=TOTAL_SLOTS),
            share_features_extractor=False, # Decouple Policy and Value networks to improve EV
            net_arch=dict(pi=[256, 256], vf=[512, 512, 512, 256])
        )
    else:
        print("Using Architecture: MLP (Standard Dense Layers)")
        params["policy_kwargs"] = dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]))
        
    return params
