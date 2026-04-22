import os
import json

# --- Directory Configuration ---
MODEL_DIR = os.getenv("MODEL_DIR", "backend/models")
REPORT_DIR = os.getenv("REPORT_DIR", "reports")
EVAL_DIR = os.path.join(REPORT_DIR, "evaluations")
LOG_DIR = os.getenv("LOG_DIR", "ppo_trading_tensorboard")

# --- Training Configuration ---
TOTAL_TIMESTEPS = 5_000_000 # Increased for vectorized environment depth
INITIAL_CAPITAL = 1_000_000.0
MIN_SYMBOLS = 5
MAX_SYMBOLS = 6
TOTAL_SLOTS = 6
FIXED_COMMISSION = 20.0 # Per order side (Upstox)

# --- Reward Tuning ---
PATIENCE_BONUS = 0.0005         # Per-step bonus for holding flat (no position, chose Hold)
DRAWDOWN_THRESHOLD_SOFT = 0.06   # Early warning DD penalty (6%)
DRAWDOWN_THRESHOLD_HARD = 0.12   # Severe exit-forcing DD penalty (12%)
SOFT_PENALTY_SCALE = 0.5         # Gentle slope for soft penalty
HARD_PENALTY_SCALE = 2.0         # Sharp slope for hard penalty
PENNY_JUNK_PENALTY = 0.005      # Penalty for trades < ₹2.0 on expiry day
PENNY_THRESHOLD = 5.0           # Threshold for junk option detection
MIN_OPTION_PRICE = 5.0          # Minimum price to allow trade entry

# Graduated commission tiers: {max_hold_duration: multiplier}
# Scalps (1 step) pay 5× base, long holds (10+) pay 0.5× base
COMMISSION_TIERS = [
    (1, 7.0),    # 1 step   → 7.0× commission (Penalize scalps harder)
    (2, 3.0),    # 2 steps  → 3.0× commission
    (4, 1.5),    # 3-4 steps → 1.5× commission
    (9, 1.0),    # 5-9 steps → 1.0× commission
]
COMMISSION_TIER_DEFAULT = 0.5    # 10+ steps → 0.5× commission

# --- Volatility Position Sizing ---
VOL_SCALE_HIGH_THRESHOLD = 0.40   # Above this vol → 50% size
VOL_SCALE_MED_THRESHOLD = 0.25    # Above this vol → 75% size

# --- Minimum Hold Period ---
MIN_HOLD_STEPS = 3               # Minimum steps before discretionary exit is allowed

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

# 2 symbols for Zero-Shot Robustness
ZERO_SHOT_SYMBOLS = [
    "SENSEX", "Nifty IT"
]

# Evaluation-only (Hidden during training)
EVAL_ONLY_SYMBOLS = [
    "SENSEX", "Nifty Bank", "Nifty 50"
]

# --- Risk Management Categories ---
# SL/TP mapped as (StopLossPct, TakeProfitPct)
# 0: No SL/TP (Full discretionary)
# 1: Tight (10% SL, 25% TP)
# 2: Regular (20% SL, 50% TP)
# 3: Aggressive (35% SL, 150% TP)
SL_TP_CATEGORIES = [
    (0.0, 0.0),    
    (0.10, 0.25),  
    (0.20, 0.50),  
    (0.35, 1.50)   
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
        "learning_rate": 1e-4,
        "batch_size": 1024,
        "n_steps": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,
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
            features_extractor_kwargs=dict(features_dim=512, lookback_window=30, total_slots=TOTAL_SLOTS),
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
    else:
        print("Using Architecture: MLP (Standard Dense Layers)")
        params["policy_kwargs"] = dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]))
        
    return params
