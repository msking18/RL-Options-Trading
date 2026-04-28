import os
import optuna
from datetime import datetime
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from backend.env.trading_env import TradingEnv
from backend.train.train_ppo import make_env
from backend.train.evaluate_ppo import evaluate_regime
from backend.train.ppo_config import (
    TRAINING_SYMBOLS, INITIAL_CAPITAL, MODEL_DIR, LOG_DIR, 
    MIN_SYMBOLS, MAX_SYMBOLS, TOTAL_SLOTS, FIXED_COMMISSION, LOOKBACK_WINDOW
)
from backend.train.feature_extractors import Trading1DCNN

GLOBAL_TENSOR = None
GLOBAL_COL_MAP = None

def load_global_data():
    """Loads and vectorizes data once for the entire study."""
    global GLOBAL_TENSOR, GLOBAL_COL_MAP
    if GLOBAL_TENSOR is None:
        print("\n--- Pre-loading and Vectorizing Data for Tuning Study ---")
        temp_env = TradingEnv(symbols=TRAINING_SYMBOLS)
        GLOBAL_TENSOR = temp_env.data_tensor
        GLOBAL_COL_MAP = temp_env.col_to_idx
    return GLOBAL_TENSOR, GLOBAL_COL_MAP

def objective(trial):
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096])
    batch_size = 1024 # Optimized for GPU throughput
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 5e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    
    # 2. Setup Environment
    # BENCHMARK showed DummyVecEnv(1) is ~2x faster than SubprocVecEnv(4) 
    # for our ultra-fast vectorized environment on Windows.
    num_envs = 1 
    
    tensor, col_map = load_global_data()
    
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([
        make_env(
            TRAINING_SYMBOLS, LOOKBACK_WINDOW, INITIAL_CAPITAL, 0.001, 0, 
            preloaded_tensor=tensor,
            col_to_idx=col_map,
            min_symbols=MIN_SYMBOLS,
            max_symbols=MAX_SYMBOLS,
            total_slots=TOTAL_SLOTS,
            fixed_commission=FIXED_COMMISSION
        )
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # 3. Setup Model
    policy_kwargs = dict(
        features_extractor_class=Trading1DCNN,
        features_extractor_kwargs=dict(features_dim=512, lookback_window=LOOKBACK_WINDOW, total_slots=TOTAL_SLOTS),
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    
    model = MaskablePPO(
        "MlpPolicy", # We use our custom extractor via policy_kwargs
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=LOG_DIR,
        device = 'auto' # Use CUDA if available (user has NVIDIA GPU)
    )
    
    # 4. Train
    tuning_timesteps = 300_000
    model.learn(
        total_timesteps=tuning_timesteps,
        tb_log_name=f"trial_{trial.number}",
        progress_bar=True
    )
    
    # 5. Evaluate on 1-Week High Volatility Benchmark
    # High Vol: 2026-03-08 to 2026-03-15
    try:
        # Save temp stats for eval
        temp_stats_path = f"tuning_stats_trial_{trial.number}.pkl"
        env.save(temp_stats_path)
        
        eval_result = evaluate_regime(
            model, "Tuning_Eval", TRAINING_SYMBOLS, 
            "2026-03-08", "2026-03-15",
            # evaluate_regime will load from DB as needed
            stats_path=temp_stats_path
        )
        
        metrics = eval_result['metrics']
        # Fix: Sync keys with backend/train/metrics.py
        total_return = metrics.get('total_return_pct', -100)
        sharpe = metrics.get('sharpe_ratio', 0)
        
        # Combined Score: Priority on surviving (sharpe) + making money
        score = (0.5 * total_return) + (0.5 * max(0, sharpe) * 10) 
        
        # Cleanup
        env.close() # CRITICAL: Close subprocesses
        if os.path.exists(temp_stats_path):
            os.remove(temp_stats_path)
        
        return score
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        if 'env' in locals():
            env.close() # CRITICAL: Close subprocesses on failure
        return -1000 

def main():
    # Load data once in the main process
    load_global_data()
    
    print("\n--- Starting Optuna Hyperparameter Tuning ---")
    study_name = f"ppo_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    study.optimize(objective, n_trials=20)
    
    print("\n--- Tuning Complete ---")
    print(f"Best Trial Score: {study.best_value}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Save best params to a JSON file in the models directory
    import json
    best_params_path = os.path.join(MODEL_DIR, "best_hyperparams.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
