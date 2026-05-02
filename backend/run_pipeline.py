import os
import subprocess
import argparse
from datetime import datetime
from backend.train.ppo_config import EVAL_DIR, REPORT_DIR

def run_command(command):
    print(f"Executing: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with exit code {process.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Master Pipeline Orchestrator for RL Options Trading")
    parser.add_argument("--timesteps", type=int, default=2000000, help="Total timesteps for training")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training phase")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation phase")
    parser.add_argument("--skip-report", action="store_true", help="Skip the reporting phase")
    parser.add_argument("--force-fresh", action="store_true", help="Force training from scratch")
    parser.add_argument("--model-source", type=str, default="best_ev",
                        choices=["best_ev", "early_stop", "latest"],
                        help="Which checkpoint to evaluate after training")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== Pipeline Run: {timestamp} ===")

    # 1. Train
    if not args.skip_train:
        print("\n--- Phase 1: Training ---")
        train_cmd = [
            "python", "-m", "backend.train.train_ppo",
            "--total-timesteps", str(args.timesteps),
            "--timestamp", timestamp
        ]
        if args.force_fresh:
            train_cmd.append("--force-fresh")
        if not run_command(train_cmd):
            return

    # 2. Evaluate
    if not args.skip_eval:
        print("\n--- Phase 2: Evaluation ---")
        eval_cmd = [
            "python", "-m", "backend.train.evaluate_ppo",
            "--timestamp", timestamp,
            "--model-source", args.model_source
        ]
        if not run_command(eval_cmd):
            return

    # 3. Report
    if not args.skip_report:
        print("\n--- Phase 3: Reporting ---")
        # Generate Excel Report
        report_cmd = [
            "python", "-m", "backend.reporting.excel_reporting",
            "--input", os.path.join(EVAL_DIR, f"eval_results_{timestamp}.pkl"),
            "--output", os.path.join(REPORT_DIR, f"portfolio_report_{timestamp}.xlsx")
        ]
        if not run_command(report_cmd):
            return

    print(f"\n=== Pipeline Completed Successfully [{timestamp}] ===")

if __name__ == "__main__":
    main()
