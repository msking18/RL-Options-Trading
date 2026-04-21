import os
import argparse
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize
from backend.env.trading_env import TradingEnv
from backend.train.metrics import (
    calculate_performance_metrics, BuyAndHoldBenchmark, 
    StraddleSellerBenchmark, MovingAverageCrossoverBenchmark,
    AlwaysATMStraddleBenchmark
)
from backend.train.ppo_config import (
    MODEL_DIR, EVAL_DIR, TRAINING_SYMBOLS, ZERO_SHOT_SYMBOLS, get_ppo_params,
    TOTAL_SLOTS, MIN_SYMBOLS, MAX_SYMBOLS, INITIAL_CAPITAL,
    FIXED_COMMISSION, VAL_START_DATE, VAL_END_DATE
)

def get_raw_env(env):
    """Recursively unwraps VecEnv to get the internal TradingEnv."""
    current = env
    while True:
        if hasattr(current, 'venv'):
            current = current.venv
        elif hasattr(current, 'envs'):
            current = current.envs[0]
        else:
            break
    return current

def preload_symbols_data(symbols, start_date, end_date):
    """
    Pre-loads and pre-processes all data for symbols across the full date range.
    Returns a dict of symbol -> pre-processed DataFrame.
    """
    print(f"\n--- Pre-loading Data for Caching ({start_date} to {end_date}) ---")
    # Buffer to ensure lookback window is available for the first regime
    buffer_start = (pd.to_datetime(start_date) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    
    # Temporary env to trigger loading and pre-processing
    temp_env = TradingEnv(
        symbols=symbols,
        start_date=buffer_start,
        end_date=end_date,
        min_active_symbols=1 # Allow pre-loading even for single symbols or small tracks
    )
    
    return temp_env.all_dfs

def evaluate_regime(model, name, symbols, start_date, end_date, preloaded_data=None, stats_path=None):
    """Runs the model and benchmarks on a specific date range for the entire portfolio."""
    print(f"\n--- Evaluating Regime: {name} ({start_date} to {end_date}) ---")
    
    regime_data = {}
    if preloaded_data:
        sd = pd.to_datetime(start_date).tz_localize(None)
        ed = pd.to_datetime(end_date).tz_localize(None)
        
        for sym, df in preloaded_data.items():
            # Ensure df['parsed_time'] is naive for comparison
            df_time = df['parsed_time'].dt.tz_localize(None)
            
            # Filter and ensure lookback window is present
            start_indices = df.index[df_time >= sd].tolist()
            if not start_indices:
                print(f"Warning: No data for {sym} in regime {name} after {sd}")
                continue
            
            # TradingEnv starts at lookback_window, so we need 30 steps before sd
            start_idx = max(0, start_indices[0] - 30)
            
            end_matches = df.index[df_time <= ed].tolist()
            if not end_matches:
                print(f"Warning: End date {ed} out of range for {sym} in regime {name}")
                continue
            end_idx = end_matches[-1]
            regime_data[sym] = df.iloc[start_idx : end_idx + 1].copy()

    # Initialize Joint Env for evaluation
    env = TradingEnv(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        slippage=0.0005,
        preloaded_data=regime_data if regime_data else None,
        min_active_symbols=min(len(symbols), MIN_SYMBOLS) if symbols else MIN_SYMBOLS,
        max_active_symbols=MAX_SYMBOLS,
        total_slots=TOTAL_SLOTS,
        fixed_commission=FIXED_COMMISSION
    )
    
    # Always wrap in DummyVecEnv for consistent step returns
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    if stats_path and os.path.exists(stats_path):
        print(f"Applying normalization stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False 
    elif stats_path:
        print(f"Warning: stats_path {stats_path} not found. Evaluation may be inaccurate.")
    
    # 1. Evaluate Portfolio Model
    raw_env = get_raw_env(env)
    obs = env.reset() # VecEnv reset returns only obs
    done = False
    equity_curve = [raw_env.state_manager.initial_capital]
    actions_log = []
    final_trade_logs = []
    
    while not done:
        action_masks = np.array([raw_env.action_masks()])
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done_tuple, info_tuple = env.step(action) # VecEnv returns 4 values
        done = done_tuple[0]
        info = info_tuple[0]
        
        equity_curve.append(info['capital'])
        actions_log.append(action)
        
        if done:
            # Capture trade logs from info before VecEnv auto-resets the environment
            final_trade_logs = info.get('trade_logs', [])
            
    model_metrics = calculate_performance_metrics(equity_curve)
    
    # 2. Evaluate Portfolio Benchmarks
    available_symbols = [s for s in symbols if s in raw_env.all_dfs]
    if not available_symbols:
        print(f"Warning: No symbols with data available for regime {name}")
        min_len = len(equity_curve)
        portfolio_bh_curves = [[INITIAL_CAPITAL] * min_len]
        portfolio_ss_curves = [[INITIAL_CAPITAL] * min_len]
        portfolio_ma_curves = [[INITIAL_CAPITAL] * min_len]
        portfolio_straddle_long_curves = [[INITIAL_CAPITAL] * min_len]
    else:
        portfolio_bh_curves = []
        portfolio_ss_curves = []
        portfolio_ma_curves = []
        portfolio_straddle_long_curves = []
        
        for sym in available_symbols:
            df = raw_env.all_dfs[sym]
            # Use the length of the agent's equity curve to determine the benchmark price slice.
            # This is robust against VecEnv auto-resets clearing current_step.
            bench_len = len(equity_curve)
            prices = df['close'].values[raw_env.lookback_window : raw_env.lookback_window + bench_len]
            call_prices = df['e0_call_price'].values[raw_env.lookback_window : raw_env.lookback_window + bench_len]
            put_prices = df['e0_put_price'].values[raw_env.lookback_window : raw_env.lookback_window + bench_len]
            
            if len(prices) == 0:
                print(f"  Warning: prices length is 0 for {sym} (bench_len={bench_len}, lookback={raw_env.lookback_window})")
                continue
            
            print(f"  Benchmark Data ({sym}): len={len(prices)}, first={prices[0]:.2f}, last={prices[-1]:.2f}, diff={prices[-1]-prices[0]:.2f}")
                
            bh_bench = BuyAndHoldBenchmark(initial_capital=INITIAL_CAPITAL)
            portfolio_bh_curves.append(bh_bench.evaluate(prices))
            
            ss_bench = StraddleSellerBenchmark(initial_capital=INITIAL_CAPITAL)
            portfolio_ss_curves.append(ss_bench.evaluate(prices))
            
            ma_bench = MovingAverageCrossoverBenchmark(initial_capital=INITIAL_CAPITAL)
            portfolio_ma_curves.append(ma_bench.evaluate(prices))
            
            straddle_long_bench = AlwaysATMStraddleBenchmark(initial_capital=INITIAL_CAPITAL)
            portfolio_straddle_long_curves.append(straddle_long_bench.evaluate(call_prices, put_prices))
        
        if not portfolio_bh_curves:
            min_len = len(equity_curve)
        else:
            min_len = min(len(equity_curve), min(len(c) for c in portfolio_bh_curves))
    
    # Final Alignment and Metric Calculation
    if min_len < 1:
        print(f"Warning: min_len is 0 for regime {name}. Forcing length 1.")
        min_len = 1
        
    equity_curve = equity_curve[:min_len]
    
    # Calculate means only if we have data
    if available_symbols and portfolio_bh_curves:
        bh_curve = np.mean([c[:min_len] for c in portfolio_bh_curves], axis=0).tolist()
        ss_curve = np.mean([c[:min_len] for c in portfolio_ss_curves], axis=0).tolist()
        ma_curve = np.mean([c[:min_len] for c in portfolio_ma_curves], axis=0).tolist()
        straddle_long_curve = np.mean([c[:min_len] for c in portfolio_straddle_long_curves], axis=0).tolist()
    else:
        # Fallback to horizontal initial capital line
        bh_curve = ss_curve = ma_curve = straddle_long_curve = [INITIAL_CAPITAL] * min_len
    
    # Final model metrics calculated on the aligned curve
    model_metrics = calculate_performance_metrics(equity_curve)
    bh_metrics = calculate_performance_metrics(bh_curve)
    ss_metrics = calculate_performance_metrics(ss_curve)
    ma_metrics = calculate_performance_metrics(ma_curve)
    straddle_long_metrics = calculate_performance_metrics(straddle_long_curve)
    
    # Print Comparison
    df_results = pd.DataFrame({
        "Metric": list(model_metrics.keys()),
        "Portfolio Agent": list(model_metrics.values()),
        "Mean B&H": list(bh_metrics.values()),
        "EMA Crossover": list(ma_metrics.values()),
        "Long Straddle": list(straddle_long_metrics.values())
    })
    print("\n" + df_results.to_string(index=False))
    
    return {
        "name": name,
        "equity_curve": equity_curve,
        "bh_curve": bh_curve,
        "ss_curve": ss_curve,
        "ma_curve": ma_curve,
        "straddle_long_curve": straddle_long_curve,
        "actions": actions_log,
        "trade_logs": final_trade_logs,
        "metrics": model_metrics,
        "symbols": symbols
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = args.model_path if args.model_path else os.path.join(MODEL_DIR, "ppo_latest.zip")
    stats_path = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    symbols_list = args.symbols.split(",") if args.symbols else TRAINING_SYMBOLS
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model for portfolio evaluation: {model_path}")
    device = get_ppo_params().get('device', 'cpu')
    model = MaskablePPO.load(model_path, device=device)
    
    # Define Evaluation Tracks
    evaluation_tracks = [
        {
            "name": "Standard OoS (6-Month)", 
            "symbols": TRAINING_SYMBOLS, 
            "start": "2025-10-01", 
            "end": "2026-04-10"
        },
        {
            "name": "Low Volatility (1-Month)", 
            "symbols": TRAINING_SYMBOLS, 
            "start": "2026-01-01", 
            "end": "2026-01-31"
        },
        {
            "name": "High Volatility (1-Week)", 
            "symbols": TRAINING_SYMBOLS, 
            "start": "2026-04-03", 
            "end": "2026-04-10"
        },
        {
            "name": "Zero-Shot Transfer (Sector Change)", 
            "symbols": ZERO_SHOT_SYMBOLS, 
            "start": "2025-10-01", 
            "end": "2026-04-10"
        }
    ]
    
    all_track_results = []
    for track in evaluation_tracks:
        try:
            # Pre-load data for this specific track's symbols to save time/memory
            track_cache = preload_symbols_data(track['symbols'], track['start'], track['end'])
            
            res = evaluate_regime(model, track['name'], track['symbols'], track['start'], track['end'], 
                                 preloaded_data=track_cache, stats_path=stats_path)
            all_track_results.append(res)
        except Exception as e:
            print(f"Error evaluating track {track['name']}: {e}")
            import traceback
            traceback.print_exc()
        
    os.makedirs(EVAL_DIR, exist_ok=True)
    results_filename = f"eval_results_{timestamp}.pkl"
    results_path = os.path.join(EVAL_DIR, results_filename)
    
    with open(results_path, "wb") as f:
        pickle.dump(all_track_results, f)
        
    latest_results_path = os.path.join(EVAL_DIR, "eval_latest.pkl")
    with open(latest_results_path, "wb") as f:
        pickle.dump(all_track_results, f)
        
    print(f"\nEvaluation complete. Results saved to {results_filename}")

if __name__ == "__main__":
    main()
