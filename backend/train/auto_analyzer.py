import os
import pickle
import pandas as pd
import numpy as np
import argparse
import shutil
import glob
from datetime import datetime
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from backend.train.ppo_config import EVAL_DIR, LOG_DIR, REPORT_DIR, TRAINING_SYMBOLS

# --- Analysis Constants ---
ANALYSIS_DIR = os.path.join(REPORT_DIR, "analysis")
BENCHMARK_KEYWORD = "(benchmark)"
PRIMARY_REGIME = "Standard OoS (6-Month)"

def find_eval_files(eval_dir):
    """Finds all eval results and identifies current, previous, and benchmark."""
    all_files = glob.glob(os.path.join(eval_dir, "eval_results_*.pkl"))
    
    # Filter out 'latest' and 'archived'
    valid_files = [f for f in all_files if "latest" not in f and "archived" not in f]
    
    benchmark = next((f for f in valid_files if BENCHMARK_KEYWORD in f), None)
    
    # Sort non-benchmark files by timestamp in filename
    timed_files = [f for f in valid_files if BENCHMARK_KEYWORD not in f]
    
    def get_timestamp(fname):
        try:
            # eval_results_20260508_205055.pkl
            base = os.path.basename(fname)
            parts = base.split("_")
            return parts[2] + "_" + parts[3].split(".")[0]
        except:
            return "00000000_000000"

    timed_files.sort(key=get_timestamp, reverse=True)
    
    current = timed_files[0] if timed_files else None
    previous = timed_files[1] if len(timed_files) > 1 else None
    
    return current, previous, benchmark

def find_matching_tb_run(tb_dir, eval_path):
    """Matches a TB run folder to an eval file based on timestamp."""
    if not eval_path: return None
    
    eval_base = os.path.basename(eval_path)
    # Extract YYYYMMDD_HHMMSS
    try:
        eval_ts_str = eval_base.split("_")[2] + "_" + eval_base.split("_")[3].split(".")[0]
        eval_ts = datetime.strptime(eval_ts_str, "%Y%m%d_%H%M%S")
    except:
        return None

    all_runs = [d for d in os.listdir(tb_dir) if os.path.isdir(os.path.join(tb_dir, d))]
    
    best_match = None
    min_diff = float('inf')
    
    for run in all_runs:
        # PPO_Portfolio_20260508_183519_1
        try:
            parts = run.split("_")
            run_ts_str = parts[2] + "_" + parts[3]
            run_ts = datetime.strptime(run_ts_str, "%Y%m%d_%H%M%S")
            
            diff = (eval_ts - run_ts).total_seconds()
            # TB run must be BEFORE or very close to eval
            if 0 <= diff < min_diff:
                min_diff = diff
                best_match = os.path.join(tb_dir, run)
        except:
            continue
            
    return best_match

def load_eval_data(path):
    if not path or not os.path.exists(path): return None
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_tb_data(run_dir):
    """Extracts key scalars from TB logs."""
    if not run_dir or not os.path.exists(run_dir): return {}
    
    print(f"Parsing TensorBoard logs: {run_dir}")
    acc = EventAccumulator(run_dir)
    acc.Reload()
    
    tags = acc.Tags().get('scalars', [])
    data = {}
    
    target_tags = [
        'train/entropy_loss', 'train/clip_fraction', 'train/explained_variance',
        'train/value_loss', 'train/policy_gradient_loss', 'train/approx_kl',
        'rollout/ep_rew_mean', 'rollout/ep_len_mean'
    ]
    
    for tag in target_tags:
        if tag in tags:
            events = acc.Scalars(tag)
            data[tag] = [(e.step, e.value) for e in events]
            
    return data

def analyze_regime_metrics(current_regime, benchmark_regime, previous_regime):
    """Compares metrics for a single regime."""
    curr_m = current_regime.get('metrics', {})
    bench_m = benchmark_regime.get('metrics', {}) if benchmark_regime else {}
    prev_m = previous_regime.get('metrics', {}) if previous_regime else {}
    
    metrics_to_compare = [
        'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct', 'profit_factor'
    ]
    
    comparison = {}
    for m in metrics_to_compare:
        c_val = curr_m.get(m, 0)
        b_val = bench_m.get(m, 0)
        p_val = prev_m.get(m, 0)
        
        comparison[m] = {
            'current': c_val,
            'benchmark': b_val,
            'previous': p_val,
            'diff_bench': c_val - b_val,
            'diff_prev': c_val - p_val
        }
        
    return comparison

def analyze_trades(trade_logs):
    """Computes trade behavior statistics."""
    if not trade_logs:
        return {"total_trades": 0, "avg_hold": 0, "avg_pnl": 0, "profit_factor": 0}
        
    pnls = [t.get('pnl_pct', 0) for t in trade_logs]
    holds = [t.get('hold_duration', 0) for t in trade_logs]
    
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    
    return {
        "total_trades": len(trade_logs),
        "avg_hold": np.mean(holds) if holds else 0,
        "avg_pnl": np.mean(pnls) * 100 if pnls else 0,
        "profit_factor": sum(wins) / sum(losses) if losses and sum(losses) > 0 else (float('inf') if wins else 1.0),
        "win_rate": len(wins) / len(trade_logs) if trade_logs else 0
    }

def check_promotion(current_data, benchmark_data):
    """
    Promotion Criteria:
    1. Majority of regimes win on Sharpe (>=3/4 or >=50% if fewer)
    2. Primary regime must win on return
    3. No catastrophic drawdown (>25% relative increase)
    4. Primary regime > 5 trades
    """
    if not benchmark_data: return True, "Initial benchmark set."
    
    regimes_count = len(current_data)
    sharpe_wins = 0
    primary_win_return = False
    dd_ok = True
    primary_trades_ok = False
    
    bench_dict = {r['name']: r for r in benchmark_data}
    
    for res in current_data:
        name = res['name']
        bench_res = bench_dict.get(name)
        if not bench_res: continue
        
        curr_m = res['metrics']
        bench_m = bench_res['metrics']
        
        if curr_m.get('sharpe_ratio', 0) >= bench_m.get('sharpe_ratio', 0):
            sharpe_wins += 1
            
        if name == PRIMARY_REGIME:
            if curr_m.get('total_return_pct', 0) > bench_m.get('total_return_pct', 0):
                primary_win_return = True
            if len(res.get('trade_logs', [])) >= 5:
                primary_trades_ok = True
                
        # Drawdown check
        curr_dd = curr_m.get('max_drawdown_pct', 0)
        bench_dd = bench_m.get('max_drawdown_pct', 0)
        if curr_dd > bench_dd * 1.25 and curr_dd > 5.0: # Allow some slack for very low DDs
            dd_ok = False
            
    promotion_ready = (sharpe_wins >= max(2, regimes_count // 2 + 1)) and primary_win_return and dd_ok and primary_trades_ok
    
    reason = []
    if not promotion_ready:
        if sharpe_wins < max(2, regimes_count // 2 + 1): reason.append("Sharpe majority not met")
        if not primary_win_return: reason.append("Primary regime return lower than benchmark")
        if not dd_ok: reason.append("Drawdown exceeded safety threshold")
        if not primary_trades_ok: reason.append("Insufficient trades in primary regime")
        
    return promotion_ready, ", ".join(reason)

def perform_promotion(current_path, benchmark_path):
    """Archives old benchmark and sets new one."""
    if not os.path.exists(current_path): return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if benchmark_path and os.path.exists(benchmark_path):
        # Archive old
        arch_name = benchmark_path.replace("(benchmark).pkl", f"(archived_{timestamp}).pkl")
        os.rename(benchmark_path, arch_name)
        print(f"Archived old benchmark: {os.path.basename(arch_name)}")
        
    # Set new
    new_bench_path = os.path.join(EVAL_DIR, f"eval_results_{timestamp}(benchmark).pkl")
    shutil.copy(current_path, new_bench_path)
    print(f"🏆 NEW BENCHMARK SET: {os.path.basename(new_bench_path)}")
    return new_bench_path

def generate_report(current_eval_path, analysis_data):
    """Generates the markdown report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"analysis_{timestamp}.md"
    report_path = os.path.join(ANALYSIS_DIR, report_name)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"# RL Evaluation Analysis — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Promotion Status
        if analysis_data.get('promoted'):
            f.write("> [!IMPORTANT]\n")
            f.write("> 🏆 **NEW BENCHMARK SET**: This run definitively beat the previous benchmark and has been promoted.\n\n")
            
        # 1. Executive Summary
        f.write("## 🎯 Executive Summary\n")
        summary = analysis_data['summary']
        f.write(f"{summary}\n\n")
        
        # 2. Metrics Table
        f.write("## 📊 Metrics Comparison\n\n")
        f.write("| Regime | Metric | Current | Benchmark | Δ Bench | Previous | Δ Prev |\n")
        f.write("|--------|--------|---------|-----------|---------|----------|--------|\n")
        
        for regime_name, metrics in analysis_data['regime_comparisons'].items():
            for m_name, vals in metrics.items():
                diff_b = f"{vals['diff_bench']:+.2f}"
                diff_p = f"{vals['diff_prev']:+.2f}"
                f.write(f"| {regime_name} | {m_name} | {vals['current']:.2f} | {vals['benchmark']:.2f} | {diff_b} | {vals['previous']:.2f} | {diff_p} |\n")
        f.write("\n")
        
        # 3. Training Stability
        f.write("## 📈 Training Stability (TensorBoard)\n")
        tb = analysis_data['tb_analysis']
        if tb:
            f.write(f"- **Clip Fraction**: {tb['final_clip']:.4f} " + ("⚠️ (High)" if tb['final_clip'] > 0.3 else "✅") + "\n")
            f.write(f"- **Entropy**: {tb['start_entropy']:.4f} → {tb['final_entropy']:.4f} (Decay: {tb['entropy_decay']:.1f}%)\n")
            f.write(f"- **Explained Variance**: {tb['final_ev']:.4f} " + ("✅" if tb['final_ev'] > 0.5 else "⚠️ (Low)") + "\n")
            f.write(f"- **Value Loss**: {tb['final_val_loss']:.4f}\n\n")
        else:
            f.write("*No TensorBoard data found for this run.*\n\n")
            
        # 4. Trade Behavior
        f.write("## 🔄 Trade Behavior\n")
        trades = analysis_data['trade_stats']
        f.write(f"- **Total Trades**: {trades['total_trades']} | **Avg Hold**: {trades['avg_hold']:.1f} steps\n")
        f.write(f"- **Win Rate**: {trades['win_rate']*100:.1f}% | **Profit Factor**: {trades['profit_factor']:.2f}\n")
        f.write(f"- **Avg PnL**: {trades['avg_pnl']:.2f}%\n\n")
        
        # 5. Worked / Failed / Suggested
        f.write("## ✅ What Worked\n")
        for item in analysis_data['worked']: f.write(f"- {item}\n")
        if not analysis_data['worked']: f.write("- No significant improvements detected.\n")
        
        f.write("\n## ❌ What Didn't Work\n")
        for item in analysis_data['failed']: f.write(f"- {item}\n")
        if not analysis_data['failed']: f.write("- No major regressions detected.\n")
        
        f.write("\n## 🔧 Suggested Improvements\n")
        for item in analysis_data['suggestions']: f.write(f"- {item}\n")
        
    print(f"Report generated: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, help="Specific eval .pkl path")
    parser.add_argument("--watch", action="store_true", help="Watch for new files")
    args = parser.parse_args()
    
    if args.watch:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            import time
            
            class NewEvalHandler(FileSystemEventHandler):
                def on_created(self, event):
                    if event.is_directory or not event.src_path.endswith(".pkl"): return
                    if "eval_results_" in event.src_path and BENCHMARK_KEYWORD not in event.src_path:
                        print(f"\nNew eval detected: {event.src_path}")
                        time.sleep(5) # Wait for write to finish
                        run_analysis(event.src_path)
                        
            observer = Observer()
            observer.schedule(NewEvalHandler(), EVAL_DIR, recursive=False)
            observer.start()
            print(f"Watching for new evals in {EVAL_DIR}...")
            try:
                while True: time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
            observer.join()
        except ImportError:
            print("Error: watchdog not installed. Run 'pip install watchdog'.")
        return

    # Manual Run
    current, previous, benchmark = find_eval_files(EVAL_DIR)
    target = args.eval if args.eval else current
    
    if not target:
        print("No evaluation files found.")
        return
        
    run_analysis(target)

def run_analysis(eval_path):
    print(f"\nAnalyzing: {os.path.basename(eval_path)}")
    current_data = load_eval_data(eval_path)
    if not current_data: return
    
    _, previous_path, benchmark_path = find_eval_files(EVAL_DIR)
    previous_data = load_eval_data(previous_path)
    benchmark_data = load_eval_data(benchmark_path)
    
    tb_run = find_matching_tb_run(LOG_DIR, eval_path)
    tb_data = parse_tb_data(tb_run)
    
    analysis = {
        'regime_comparisons': {},
        'tb_analysis': {},
        'trade_stats': {},
        'worked': [],
        'failed': [],
        'suggestions': [],
        'promoted': False
    }
    
    # 1. Regimes
    bench_dict = {r['name']: r for r in benchmark_data} if benchmark_data else {}
    prev_dict = {r['name']: r for r in previous_data} if previous_data else {}
    
    for res in current_data:
        name = res['name']
        analysis['regime_comparisons'][name] = analyze_regime_metrics(
            res, bench_dict.get(name), prev_dict.get(name)
        )
        
    # 2. TB Analysis
    if tb_data:
        entropy = tb_data.get('train/entropy_loss', [(0,0)])
        clip = tb_data.get('train/clip_fraction', [(0,0)])
        ev = tb_data.get('train/explained_variance', [(0,0)])
        val_loss = tb_data.get('train/value_loss', [(0,0)])
        
        analysis['tb_analysis'] = {
            'start_entropy': entropy[0][1],
            'final_entropy': entropy[-1][1],
            'entropy_decay': (1 - entropy[-1][1]/entropy[0][1])*100 if entropy[0][1] != 0 else 0,
            'final_clip': clip[-1][1],
            'final_ev': ev[-1][1],
            'final_val_loss': val_loss[-1][1]
        }
        
    # 3. Trade Stats (Primary Regime)
    primary = next((r for r in current_data if r['name'] == PRIMARY_REGIME), current_data[0])
    analysis['trade_stats'] = analyze_trades(primary.get('trade_logs', []))
    
    # 4. Diagnosis
    comp = analysis['regime_comparisons'].get(PRIMARY_REGIME, {})
    if comp:
        if comp['sharpe_ratio']['diff_bench'] > 0: analysis['worked'].append("Improved Sharpe ratio vs. benchmark")
        else: analysis['failed'].append("Sharpe ratio below benchmark")
        
        if comp['total_return_pct']['diff_bench'] > 0: analysis['worked'].append("Higher total return vs. benchmark")
        else: analysis['failed'].append("Total return below benchmark")
        
        if comp['max_drawdown_pct']['diff_bench'] < 0: analysis['worked'].append("Reduced max drawdown vs. benchmark")
        
    # 5. Suggestions
    if analysis['tb_analysis'].get('final_clip', 0) > 0.35:
        analysis['suggestions'].append("High clip fraction: Consider reducing learning_rate or clip_range.")
    if analysis['trade_stats']['total_trades'] < 10:
        analysis['suggestions'].append("Low trade count: Check entry_penalty or increase exploration (entropy).")
    if analysis['tb_analysis'].get('final_ev', 0) < 0.1:
        analysis['suggestions'].append("Low Explained Variance: Policy is not learning from value function. Check reward scale.")
    if comp.get('max_drawdown_pct', {}).get('current', 0) > 20:
        analysis['suggestions'].append("High Drawdown: Agent is losing significant capital. Consider increasing drawdown penalty or tightening stop losses.")
        
    # 6. Promotion
    promoted, reason = check_promotion(current_data, benchmark_data)
    if promoted:
        perform_promotion(eval_path, benchmark_path)
        analysis['promoted'] = True
        analysis['summary'] = "✅ **VERDICT**: Success. This run outperformed the benchmark and has been promoted."
    else:
        analysis['summary'] = f"❌ **VERDICT**: Regression or Neutral. Did not meet promotion criteria ({reason})."
        
    generate_report(eval_path, analysis)

if __name__ == "__main__":
    main()
