#!/usr/bin/env python3
"""
parameter_optimizer.py - Test one channel against multiple parameter combinations

This script runs a grid search across all parameter combinations to find
the optimal strategy settings for a specific channel's signals.

Usage:
    python parameter_optimizer.py                           # Run all combinations
    python parameter_optimizer.py --quick                   # Run smaller test set
    python parameter_optimizer.py --params custom.json      # Use custom parameters
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from itertools import product

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtester.backtest_engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ParameterOptimizer:
    """
    Runs grid search optimization across parameter combinations.
    Tests one channel's signals with different strategy configurations.
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Define parameter grid
        if quick_mode:
            self.param_grid = self.get_quick_grid()
        else:
            self.param_grid = self.get_full_grid()
        
        self.results = []
    
    def get_quick_grid(self):
        """Smaller grid for quick testing (fewer combinations)."""
        return {
            "breakeven_trigger_percent": [10, 15],
            "trail_method": ["atr", "pullback_percent"],
            "pullback_percent": [10],
            "atr_period": [14],
            "atr_multiplier": [1.5],
            "psar_enabled": [True, False],
            "psar_start": [0.02],
            "psar_increment": [0.02],
            "psar_max": [0.2],
            "rsi_hook_enabled": [True, False],
            "rsi_period": [14],
            "rsi_overbought": [70],
            "rsi_oversold": [30]
        }
    
    def get_full_grid(self):
        """Complete grid for thorough testing (many combinations)."""
        return {
            "breakeven_trigger_percent": [5, 10, 15, 20],
            "trail_method": ["atr", "pullback_percent"],
            "pullback_percent": [8, 10, 12, 15],
            "atr_period": [10, 14, 20],
            "atr_multiplier": [1.0, 1.5, 2.0, 2.5],
            "psar_enabled": [True, False],
            "psar_start": [0.01, 0.02, 0.03],
            "psar_increment": [0.01, 0.02, 0.03],
            "psar_max": [0.1, 0.2, 0.3],
            "rsi_hook_enabled": [True, False],
            "rsi_period": [10, 14, 20],
            "rsi_overbought": [65, 70, 75],
            "rsi_oversold": [25, 30, 35]
        }
    
    def load_custom_grid(self, custom_file):
        """Loads custom parameter grid from JSON file."""
        with open(custom_file, 'r') as f:
            self.param_grid = json.load(f)
        logging.info(f"Loaded custom parameter grid from {custom_file}")
    
    def generate_combinations(self):
        """Generates all parameter combinations from the grid."""
        # Get keys and values
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        # Generate all combinations
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        logging.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_test(self, test_num, params, total_tests):
        """Runs a single backtest with specific parameters."""
        test_name = f"test_{test_num:04d}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Breakeven: {params['breakeven_trigger_percent']}%")
        logging.info(f"  Trail: {params['trail_method']}")
        logging.info(f"  PSAR: {params['psar_enabled']} | RSI: {params['rsi_hook_enabled']}")
        
        try:
            # Create backtest engine
            engine = BacktestEngine(
                signal_file_path=str(self.signals_file),
                data_folder_path="backtester/historical_data"
            )
            
            # Apply parameters to config
            profile = engine.config.profiles[0]
            
            profile['exit_strategy']['breakeven_trigger_percent'] = params['breakeven_trigger_percent']
            profile['exit_strategy']['trail_method'] = params['trail_method']
            profile['exit_strategy']['trail_settings']['pullback_percent'] = params['pullback_percent']
            profile['exit_strategy']['trail_settings']['atr_period'] = params['atr_period']
            profile['exit_strategy']['trail_settings']['atr_multiplier'] = params['atr_multiplier']
            
            profile['exit_strategy']['momentum_exits']['psar_enabled'] = params['psar_enabled']
            profile['exit_strategy']['momentum_exits']['psar_settings']['start'] = params['psar_start']
            profile['exit_strategy']['momentum_exits']['psar_settings']['increment'] = params['psar_increment']
            profile['exit_strategy']['momentum_exits']['psar_settings']['max'] = params['psar_max']
            
            profile['exit_strategy']['momentum_exits']['rsi_hook_enabled'] = params['rsi_hook_enabled']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['period'] = params['rsi_period']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['overbought_level'] = params['rsi_overbought']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['oversold_level'] = params['rsi_oversold']
            
            # Run simulation (suppress most logging)
            import logging as log
            log.getLogger().setLevel(logging.WARNING)
            engine.run_simulation()
            log.getLogger().setLevel(logging.INFO)
            
            # Analyze results
            results_file = Path("backtester/historical_data/backtest_results.csv")
            if results_file.exists():
                summary = self.analyze_results(results_file, test_name, params)
                self.results.append(summary)
                return summary
            else:
                logging.warning(f"No results file generated for {test_name}")
                return None
                
        except Exception as e:
            logging.error(f"Test {test_name} failed: {e}")
            return None
    
    def analyze_results(self, results_file, test_name, params):
        """Analyzes backtest results and generates metrics."""
        df = pd.read_csv(results_file)
        
        if df.empty:
            return {
                "test_name": test_name,
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0,
                "error": "No trades",
                **params
            }
        
        # Calculate metrics
        total_pnl = df['pnl'].sum()
        wins = df[df['pnl'] > 0]
        losses = df[df['pnl'] < 0]
        
        win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 999.99
        
        max_drawdown = (df['pnl'].cumsum().cummax() - df['pnl'].cumsum()).max()
        
        # Calculate Sharpe-like ratio (returns / volatility)
        returns_std = df['pnl'].std() if len(df) > 1 else 1
        sharpe = (df['pnl'].mean() / returns_std) if returns_std != 0 else 0
        
        return {
            "test_name": test_name,
            "total_trades": len(df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            **params
        }
    
    def generate_report(self):
        """Generates comprehensive optimization report."""
        if not self.results:
            logging.error("No results to report")
            return
        
        df = pd.DataFrame(self.results)
        
        # Sort by total P&L
        df_sorted = df.sort_values('total_pnl', ascending=False)
        
        # Save full results
        full_results_file = self.output_dir / "all_results.csv"
        df_sorted.to_csv(full_results_file, index=False)
        
        # Create summary report
        summary_file = self.output_dir / "optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PARAMETER OPTIMIZATION RESULTS\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Tests: {len(df)}\n")
            f.write(f"Signals File: {self.signals_file}\n")
            f.write("="*100 + "\n\n")
            
            # Top 10 configurations
            f.write("TOP 10 PARAMETER COMBINATIONS (by Total P&L):\n")
            f.write("-"*100 + "\n")
            for i, row in df_sorted.head(10).iterrows():
                f.write(f"\n#{i+1}. {row['test_name']} - P&L: ${row['total_pnl']:.2f}\n")
                f.write(f"   Trades: {row['total_trades']} | Win Rate: {row['win_rate']:.1f}% | Profit Factor: {row['profit_factor']:.2f}\n")
                f.write(f"   Breakeven: {row['breakeven_trigger_percent']}% | Trail: {row['trail_method']}\n")
                f.write(f"   ATR: {row['atr_multiplier']}x | Pullback: {row['pullback_percent']}%\n")
                f.write(f"   PSAR: {row['psar_enabled']} | RSI Hook: {row['rsi_hook_enabled']}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("TOP 10 BY WIN RATE:\n")
            f.write("-"*100 + "\n")
            df_by_wr = df_sorted.sort_values('win_rate', ascending=False)
            for i, row in df_by_wr.head(10).iterrows():
                f.write(f"{row['test_name']}: {row['win_rate']:.1f}% WR | ${row['total_pnl']:.2f} P&L\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("TOP 10 BY PROFIT FACTOR:\n")
            f.write("-"*100 + "\n")
            df_by_pf = df_sorted.sort_values('profit_factor', ascending=False)
            for i, row in df_by_pf.head(10).iterrows():
                f.write(f"{row['test_name']}: {row['profit_factor']:.2f} PF | ${row['total_pnl']:.2f} P&L\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("PARAMETER ANALYSIS:\n")
            f.write("-"*100 + "\n")
            
            # Analyze which parameters matter most
            f.write("\nBest Breakeven Trigger:\n")
            breakeven_analysis = df.groupby('breakeven_trigger_percent')['total_pnl'].mean().sort_values(ascending=False)
            for be, pnl in breakeven_analysis.items():
                f.write(f"  {be}%: Avg P&L ${pnl:.2f}\n")
            
            f.write("\nBest Trail Method:\n")
            trail_analysis = df.groupby('trail_method')['total_pnl'].mean().sort_values(ascending=False)
            for method, pnl in trail_analysis.items():
                f.write(f"  {method}: Avg P&L ${pnl:.2f}\n")
            
            f.write("\nBest ATR Multiplier:\n")
            atr_analysis = df.groupby('atr_multiplier')['total_pnl'].mean().sort_values(ascending=False)
            for mult, pnl in atr_analysis.items():
                f.write(f"  {mult}x: Avg P&L ${pnl:.2f}\n")
            
            f.write("\nPSAR Impact:\n")
            psar_analysis = df.groupby('psar_enabled')['total_pnl'].mean().sort_values(ascending=False)
            for enabled, pnl in psar_analysis.items():
                f.write(f"  {'Enabled' if enabled else 'Disabled'}: Avg P&L ${pnl:.2f}\n")
            
            f.write("\nRSI Hook Impact:\n")
            rsi_analysis = df.groupby('rsi_hook_enabled')['total_pnl'].mean().sort_values(ascending=False)
            for enabled, pnl in rsi_analysis.items():
                f.write(f"  {'Enabled' if enabled else 'Disabled'}: Avg P&L ${pnl:.2f}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("RECOMMENDED CONFIGURATION:\n")
            f.write("-"*100 + "\n")
            best = df_sorted.iloc[0]
            f.write(f"""
Best Overall Performance: {best['test_name']}
Total P&L: ${best['total_pnl']:.2f}
Win Rate: {best['win_rate']:.1f}%
Profit Factor: {best['profit_factor']:.2f}

Recommended Config for services/config.py:
----------------------------------------
"exit_strategy": {{
    "breakeven_trigger_percent": {int(best['breakeven_trigger_percent'])},
    "trail_method": "{best['trail_method']}",
    "trail_settings": {{
        "pullback_percent": {int(best['pullback_percent'])},
        "atr_period": {int(best['atr_period'])},
        "atr_multiplier": {best['atr_multiplier']}
    }},
    "momentum_exits": {{
        "psar_enabled": {str(best['psar_enabled'])},
        "psar_settings": {{"start": {best['psar_start']}, "increment": {best['psar_increment']}, "max": {best['psar_max']}}},
        "rsi_hook_enabled": {str(best['rsi_hook_enabled'])},
        "rsi_settings": {{"period": {int(best['rsi_period'])}, "overbought_level": {int(best['rsi_overbought'])}, "oversold_level": {int(best['rsi_oversold'])}}}
    }}
}}
""")
            f.write("="*100 + "\n")
        
        # Print to console
        print("\n" + open(summary_file).read())
        print(f"\n✅ Full results: {full_results_file}")
        print(f"✅ Summary: {summary_file}\n")
    
    def run_optimization(self):
        """Executes the complete optimization workflow."""
        if not self.signals_file.exists():
            logging.error(f"Signals file not found: {self.signals_file}")
            return
        
        logging.info(f"\n🚀 Starting parameter optimization")
        logging.info(f"Signals: {self.signals_file}")
        logging.info(f"Mode: {'Quick' if self.quick_mode else 'Full'}\n")
        
        # Generate combinations
        combinations = self.generate_combinations()
        total_tests = len(combinations)
        
        logging.info(f"Will run {total_tests} backtests\n")
        
        # Run all tests
        for i, params in enumerate(combinations, 1):
            self.run_single_test(i, params, total_tests)
        
        # Generate report
        self.generate_report()
        
        logging.info(f"\n✅ Optimization complete! Results in: {self.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter optimization for backtesting")
    parser.add_argument('--signals', type=str, default='backtester/signals_to_test.txt', help='Signals file path')
    parser.add_argument('--quick', action='store_true', help='Run quick test (fewer combinations)')
    parser.add_argument('--params', type=str, help='Custom parameter grid JSON file')
    
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    if args.params:
        optimizer.load_custom_grid(args.params)
    
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
