import os
import json

# --- Directory Configuration ---
MODEL_DIR = "backend/models"
REPORT_DIR = "reports"
EVAL_DIR = os.path.join(REPORT_DIR, "evaluations")
LOG_DIR = "ppo_trading_tensorboard"

# --- Training Configuration ---
TOTAL_TIMESTEPS = 5_000_000 # Increased for vectorized environment depth
INITIAL_CAPITAL = 1_000_000.0
MIN_SYMBOLS = 5
MAX_SYMBOLS = 10
TOTAL_SLOTS = 10
FIXED_COMMISSION = 20.0 # Per order side (Upstox)
USE_CNN = True         # Toggle for 1D-CNN vs MlpPolicy

# --- Symbol Configuration ---
TRAINING_SYMBOLS = [
    "Nifty 50", 
    "Nifty Bank", 
    "Nifty Fin Service",
    "Nifty Midcap Select",
    "Nifty Next 50",
    "Nifty 100",
    "Nifty 500",
    "Nifty IT",
    "Nifty Auto",
    "Nifty Pharma"
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
