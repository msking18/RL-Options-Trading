import os
import multiprocessing
import argparse
from datetime import datetime
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from backend.train.ppo_config import (
    get_ppo_params, TRAINING_SYMBOLS, MODEL_DIR, LOG_DIR, 
    TOTAL_TIMESTEPS, MIN_SYMBOLS, MAX_SYMBOLS, TOTAL_SLOTS, INITIAL_CAPITAL,
    FIXED_COMMISSION, TRAIN_START_DATE, TRAIN_END_DATE
)
from backend.env.trading_env import TradingEnv

class HyperparameterAnnealingCallback(BaseCallback):
    def __init__(self, initial_lr=3e-4, final_lr=1e-5, initial_ent=0.01, final_ent=0.0, verbose=0):
        super(HyperparameterAnnealingCallback, self).__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.initial_ent = initial_ent
        self.final_ent = final_ent

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.locals["total_timesteps"]
        progress = min(1.0, max(0.0, progress))
        
        new_lr = self.initial_lr - progress * (self.initial_lr - self.final_lr)
        new_ent = self.initial_ent - progress * (self.initial_ent - self.final_ent)
        
        self.model.ent_coef = new_ent
        
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        self.model.learning_rate = new_lr
        
        self.logger.record("config/learning_rate", new_lr)
        self.logger.record("config/ent_coef", new_ent)
        
        return True

class TradeDebugCallback(BaseCallback):
    """
    Logs trade-specific metrics (SL hit rate, TP hit rate, avg duration) 
    to TensorBoard from the environment info.
    """
    def __init__(self, verbose=0):
        super(TradeDebugCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if any environment finished an episode (info['trade_logs'] is present)
        for info in self.locals.get("infos", []):
            if "trade_logs" in info:
                logs = info["trade_logs"]
                if not logs:
                    continue
                
                # Extract metrics from trade logs
                sl_hits = sum(1 for log in logs if log.get('sl_tp_hit') and log.get('is_sl'))
                tp_hits = sum(1 for log in logs if log.get('sl_tp_hit') and log.get('is_tp'))
                total_trades = len(logs)
                durations = [log['hold_duration'] for log in logs]
                
                if total_trades > 0:
                    sl_rate = sl_hits / total_trades
                    tp_rate = tp_hits / total_trades
                    avg_dur = sum(durations) / total_trades
                    
                    self.logger.record("debug/sl_hit_rate", sl_rate)
                    self.logger.record("debug/tp_hit_rate", tp_rate)
                    self.logger.record("debug/avg_hold_duration", avg_dur)
                    self.logger.record("debug/total_trades_per_episode", total_trades)
        
        return True

def mask_fn(env):
    return env.action_masks()

def make_env(symbols, lookback_window, initial_capital, slippage, rank, seed=0, preloaded_data=None, min_symbols=5, max_symbols=6, total_slots=6, fixed_commission=0.0, preloaded_tensor=None, col_to_idx=None, start_date=None, end_date=None):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = TradingEnv(
            symbols=symbols,
            lookback_window=lookback_window,
            initial_capital=initial_capital,
            slippage=slippage,
            preloaded_data=preloaded_data,
            min_active_symbols=min_symbols,
            max_active_symbols=max_symbols,
            total_slots=total_slots,
            fixed_commission=fixed_commission,
            preloaded_tensor=preloaded_tensor,
            col_to_idx=col_to_idx,
            start_date=start_date,
            end_date=end_date
        )
        return ActionMasker(env, mask_fn)
    return _init

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None, help="Comma separated symbols or None for config default")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--force-fresh", action="store_true", help="Force training from scratch even if latest model exists")
    args = parser.parse_args()

    # Determine timestamp
    timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine symbols
    symbols = args.symbols.split(",") if args.symbols else TRAINING_SYMBOLS
    
    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Initializing {args.num_envs} Vectorized Environments with symbols: {symbols}")
    
    # Pre-load data once for all workers
    print(f"\n--- Pre-loading Data for Training Workers ({TRAIN_START_DATE} to {TRAIN_END_DATE}) ---")
    temp_env = TradingEnv(symbols=symbols, start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE)
    preloaded_data = temp_env.all_dfs
    
    # Determine vectorized environment type
    if args.num_envs > 1:
        # For multiple environments, use SubprocVecEnv with 'fork' for Linux speed 
        # (or 'spawn' if 'fork' has issues, but 'fork' is standard in Docker)
        try:
            # Use 'spawn' for Windows compatibility
            import platform
            method = 'fork' if platform.system() != 'Windows' else 'spawn'
            multiprocessing.set_start_method(method, force=True)
        except RuntimeError:
            pass
        
        env_class = SubprocVecEnv
        print(f"Using SubprocVecEnv with {args.num_envs} workers.")
    else:
        # For single environment, DummyVecEnv is much more stable and avoids multiprocessing bugs
        env_class = DummyVecEnv
        print("Using DummyVecEnv for stability.")

    # Create multiple environments
    env = env_class([
        make_env(
            symbols, 30, INITIAL_CAPITAL, 0.001, i, 
            preloaded_data=preloaded_data,
            min_symbols=MIN_SYMBOLS,
            max_symbols=MAX_SYMBOLS,
            total_slots=TOTAL_SLOTS,
            fixed_commission=FIXED_COMMISSION,
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE
        ) 
        for i in range(args.num_envs)
    ])
    
    # Apply VecNormalize for stable learning in high-dimensional finance data
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Add Monitor for TensorBoard stats
    env = VecMonitor(env)
    
    params = get_ppo_params()
    
    # Check for force-fresh flag (Recommended for this iteration)
    force_fresh = args.force_fresh 
    
    # Resumption Logic
    latest_path = os.path.join(MODEL_DIR, "ppo_latest.zip")
    stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    
    if os.path.exists(latest_path) and not force_fresh:
        print(f"\n--- Resuming from latest model found at {latest_path} ---")
        # Load VecNormalize stats if they exist
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
        model = MaskablePPO.load(latest_path, env=env, tensorboard_log=LOG_DIR, device=params.get('device', 'cpu'))
    else:
        if force_fresh:
            print("\n--- Forced Clean Start: Ignoring physical latest model ---")
        else:
            print("\n--- Starting training from scratch ---")
        model = MaskablePPO("MlpPolicy", env, **params, tensorboard_log=LOG_DIR)
    
    # Extra check to clarify policy architecture to the user
    policy_type = "CNN" if "features_extractor_class" in params["policy_kwargs"] else "Standard MLP"
    print(f"Model Initialized with architecture: {policy_type}")

    print(f"Starting PPO training for {args.total_timesteps} timesteps...")
    print(f"Target Model: {MODEL_DIR}/ppo_model_{timestamp}.zip")
    
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
        tb_log_name=f"PPO_Portfolio_{timestamp}",
        callback=[
            HyperparameterAnnealingCallback(
                initial_lr=params.get('learning_rate', 3e-4),
                initial_ent=params.get('ent_coef', 0.01)
            ),
            TradeDebugCallback()
        ]
    )
    
    model_path = os.path.join(MODEL_DIR, f"ppo_model_{timestamp}.zip")
    stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    timestamped_stats_path = os.path.join(MODEL_DIR, f"vec_normalize_{timestamp}.pkl")
    
    model.save(model_path)
    model.save(latest_path)
    
    # Save Normalization Statistics
    env.save(stats_path)
    env.save(timestamped_stats_path)
    
    print(f"Training complete. Model saved to {model_path} and stats to {stats_path}")
    env.close()

if __name__ == "__main__":
    train()
