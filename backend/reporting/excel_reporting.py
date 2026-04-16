import os
import pickle
import pandas as pd
import numpy as np
import xlsxwriter
import argparse
from datetime import datetime
from backend.train.ppo_config import EVAL_DIR, REPORT_DIR

def generate_excel_report(pkl_path, output_path):
    """Parses pkl results and generates a multi-sheet Excel report with charts."""
    if not os.path.exists(pkl_path):
        print(f"Error: Pickle file not found at {pkl_path}")
        return

    with open(pkl_path, "rb") as f:
        all_results = pickle.load(f)

    workbook = xlsxwriter.Workbook(output_path)
    
    # Formats
    header_format = workbook.add_format({
        'bold': True, 
        'bg_color': '#4F81BD', 
        'font_color': 'white',
        'border': 1,
        'align': 'center'
    })
    metric_format = workbook.add_format({'num_format': '#,##0.00', 'border': 1})
    pct_format = workbook.add_format({'num_format': '0.00%', 'border': 1})
    name_format = workbook.add_format({'bold': True, 'border': 1})

    # 1. Dashboard / Summary Sheet
    summary_sheet = workbook.add_worksheet("Portfolio Dashboard")
    summary_sheet.set_column('A:A', 25)
    summary_sheet.set_column('B:G', 15)

    row = 0
    col = 0
    summary_headers = ["Regime Name", "Total Return %", "Sharpe Ratio", "Max Drawdown %", "Win Rate %", "Final Capital"]
    for i, h in enumerate(summary_headers):
        summary_sheet.write(row, i, h, header_format)

    for i, res in enumerate(all_results):
        row = i + 1
        m = res['metrics']
        
        # Safe key retrieval with fallbacks for legacy support
        total_ret = m.get('total_return_pct', m.get('Total Return (%)', 0))
        sharpe = m.get('sharpe_ratio', m.get('Sharpe Ratio', 0))
        max_dd = m.get('max_drawdown_pct', m.get('Max Drawdown (%)', 0))
        win_rate = m.get('win_rate_pct', m.get('Win Rate (%)', 0))

        summary_sheet.write(row, 0, res['name'], name_format)
        summary_sheet.write(row, 1, total_ret / 100, pct_format)
        summary_sheet.write(row, 2, sharpe, metric_format)
        summary_sheet.write(row, 3, max_dd / 100, pct_format)
        summary_sheet.write(row, 4, win_rate / 100, pct_format)
        
        final_cap = res['equity_curve'][-1] if len(res['equity_curve']) > 0 else 0
        summary_sheet.write(row, 5, final_cap, metric_format)

    # Strategy Comparison Table (per regime)
    comp_row = len(all_results) + 4
    for res in all_results:
        summary_sheet.write(comp_row, 0, f"Benchmark Comparison: {res['name']}", header_format)
        headers = ["Strategy", "Total Return %", "Final Capital"]
        for i, h in enumerate(headers):
            summary_sheet.write(comp_row + 1, i, h, header_format)
        
        strats = [
            ("Portfolio Agent", res['equity_curve']),
            ("Buy & Hold", res['bh_curve']),
            ("Short Straddle", res['ss_curve']),
            ("EMA Crossover", res['ma_curve']),
            ("Long Straddle", res['straddle_long_curve'])
        ]
        
        for k, (name, curve) in enumerate(strats):
            if len(curve) > 0:
                ret = (curve[-1] - curve[0]) / max(1e-6, curve[0])
                final_val = curve[-1]
            else:
                ret = 0.0
                final_val = 0.0
                
            summary_sheet.write(comp_row + 2 + k, 0, name)
            summary_sheet.write(comp_row + 2 + k, 1, ret, pct_format)
            summary_sheet.write(comp_row + 2 + k, 2, final_val, metric_format)
        comp_row += 10

    # 2. Detailed Trade Logs Sheet
    log_sheet = workbook.add_worksheet("Trade Logs")
    log_sheet.set_column('A:B', 15)
    log_sheet.set_column('C:E', 12)
    log_sheet.set_column('F:I', 15)
    
    log_headers = ['Regime', 'Symbol', 'Type', 'Entry Step', 'Exit Step', 'Entry Price', 'Exit Price', 'PnL', 'PnL %']
    for c, h in enumerate(log_headers):
        log_sheet.write(0, c, h, header_format)

    log_row = 1
    for res in all_results:
        for log in res.get('trade_logs', []):
            log_sheet.write(log_row, 0, res['name'])
            log_sheet.write(log_row, 1, log['symbol'])
            log_sheet.write(log_row, 2, log['type'])
            log_sheet.write(log_row, 3, log['entry_time'])
            log_sheet.write(log_row, 4, log['exit_time'])
            log_sheet.write(log_row, 5, log['entry_price'], metric_format)
            log_sheet.write(log_row, 6, log['exit_price'], metric_format)
            log_sheet.write(log_row, 7, log['pnl'], metric_format)
            log_sheet.write(log_row, 8, log['pnl_pct'], pct_format)
            log_row += 1

    # 3. Equity Curves & Drawdowns Sheet
    curve_sheet = workbook.add_worksheet("Curves Data")
    for i, res in enumerate(all_results):
        base_col = i * 6
        headers = [f"{res['name']} - Step", "Agent", "B&H", "Short Straddle", "EMA Cross", "Long Straddle"]
        for c, h in enumerate(headers):
            curve_sheet.write(0, base_col + c, h, header_format)
        
        curves = [
            res['equity_curve'], res['bh_curve'], res['ss_curve'], 
            res['ma_curve'], res['straddle_long_curve']
        ]
        
        max_steps = len(res['equity_curve'])
        for step in range(max_steps):
            curve_sheet.write(step + 1, base_col, step)
            for c, curve in enumerate(curves):
                curve_sheet.write(step + 1, base_col + 1 + c, curve[step])

        # Equity Chart
        chart = workbook.add_chart({'type': 'line'})
        chart.set_title({'name': f'Cumulative PnL - {res["name"]}'})
        chart.set_x_axis({'name': 'Steps'})
        chart.set_y_axis({'name': 'Capital (INR)'})
        
        colors = ['#4F81BD', '#C0504D', '#9BBB59', '#8064A2', '#4BACC6']
        for c, name in enumerate(["Agent", "B&H", "Short Straddle", "EMA Cross", "Long Straddle"]):
            chart.add_series({
                'name': name,
                'categories': ['Curves Data', 1, base_col, max_steps, base_col],
                'values': ['Curves Data', 1, base_col+1+c, max_steps, base_col+1+c],
                'line': {'color': colors[c], 'width': 1.5 if c > 0 else 2.25}
            })
        
        # Insert if we have valid steps
        if max_steps > 0:
            summary_sheet.insert_chart(f'G{1 + i*18}', chart, {'x_scale': 1.5, 'y_scale': 1.5})

        # Drawdown Chart (New)
        dd_chart = workbook.add_chart({'type': 'area'})
        dd_chart.set_title({'name': f'Underwater Drawdown - {res["name"]}'})
        
        # Calculate DD curve and write to a hidden-ish area or new sheet
        # For simplicity, we'll just write it to columns further right
        dd_base_col = base_col + 30 # Offset for DD data
        curve_sheet.write(0, dd_base_col, "Step", header_format)
        curve_sheet.write(0, dd_base_col + 1, "Agent Drawdown", header_format)
        
        agent_curve = np.array(res['equity_curve'])
        rolling_max = np.maximum.accumulate(agent_curve)
        drawdown = (agent_curve - rolling_max) / rolling_max
        
        for step, dd in enumerate(drawdown):
            curve_sheet.write(step + 1, dd_base_col, step)
            curve_sheet.write(step + 1, dd_base_col + 1, dd)
            
        dd_chart.add_series({
            'name': 'Agent Drawdown',
            'categories': ['Curves Data', 1, dd_base_col, max_steps, dd_base_col],
            'values': ['Curves Data', 1, dd_base_col+1, max_steps, dd_base_col+1],
            'fill': {'color': '#C0504D', 'transparency': 50},
            'line': {'color': '#C0504D'}
        })
        summary_sheet.insert_chart(f'G{1 + i*18 + 9}', dd_chart, {'x_scale': 1.5, 'y_scale': 0.8})

    # 4. PnL Distribution Sheet (New)
    dist_sheet = workbook.add_worksheet("PnL Distribution")
    dist_row = 0
    for res in all_results:
        trades = res.get('trade_logs', [])
        if not trades: continue
        
        pnls = [t['pnl_pct'] for t in trades]
        dist_sheet.write(dist_row, 0, f"PnL Stats: {res['name']}", header_format)
        stats = pd.Series(pnls).describe()
        for i, (idx, val) in enumerate(stats.items()):
            dist_sheet.write(dist_row + 1 + i, 0, idx)
            dist_sheet.write(dist_row + 1 + i, 1, val, metric_format if "count" not in idx else None)
        
        # Mini Histogram helper (bins)
        counts, bins = np.histogram(pnls, bins=10)
        dist_sheet.write(dist_row, 3, "PnL Bin", header_format)
        dist_sheet.write(dist_row, 4, "Frequency", header_format)
        for i in range(len(counts)):
            dist_sheet.write(dist_row + 1 + i, 3, f"{bins[i]:.2%} to {bins[i+1]:.2%}")
            dist_sheet.write(dist_row + 1 + i, 4, counts[i])
            
        hist_chart = workbook.add_chart({'type': 'column'})
        hist_chart.add_series({
            'name': 'Trade PnL Distribution',
            'categories': ['PnL Distribution', dist_row+1, 3, dist_row+len(counts), 3],
            'values': ['PnL Distribution', dist_row+1, 4, dist_row+len(counts), 4],
        })
        hist_chart.set_title({'name': f'PnL Distribution - {res["name"]}'})
        dist_sheet.insert_chart(f'G{dist_row + 1}', hist_chart)
        
        dist_row += 15

    workbook.close()
    print(f"Excel report successfully generated: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.path.join(EVAL_DIR, "eval_latest.pkl"))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output if args.output else os.path.join(REPORT_DIR, f"portfolio_report_{timestamp}.xlsx")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    generate_excel_report(args.input, output_path)
