#!/usr/bin/env python3
"""
parameter_optimizer.py - Automated Parameter Grid Search for Trading Strategies
================================================================================
This script automates the heavy lifting of backtesting by:
1. Taking your signal files
2. Testing EVERY combination of parameters
3. Finding the optimal configuration
4. Generating detailed reports

Usage:
    python parameter_optimizer.py                    # Full grid search
    python parameter_optimizer.py --quick            # Quick mode (fewer tests)
    python parameter_optimizer.py --channel 1        # Specific channel
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Any

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtester.backtest_engine import BacktestEngine
from services.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ParameterOptimizer:
    """
    Automated parameter optimization for trading strategies.
    Tests all combinations and finds the best configuration.
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
        self.config = Config()
    
    def get_quick_grid(self) -> Dict[str, List[Any]]:
        """Quick grid for fast testing (16 combinations)."""
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
    
    def get_full_grid(self) -> Dict[str, List[Any]]:
        """Complete grid for thorough optimization (3,456 combinations)."""
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
    
    def load_custom_grid(self, custom_file: str):
        """Load custom parameter grid from JSON file."""
        with open(custom_file, 'r') as f:
            self.param_grid = json.load(f)
        logging.info(f"Loaded custom parameter grid from {custom_file}")
    
    def generate_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from the grid."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        logging.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    async def run_single_test(self, test_num: int, params: Dict, total_tests: int) -> Dict:
        """Run a single backtest with specific parameters."""
        test_name = f"test_{test_num:04d}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Breakeven: {params['breakeven_trigger_percent']}%")
        logging.info(f"  Trail: {params['trail_method']}")
        logging.info(f"  PSAR: {params['psar_enabled']} | RSI: {params['rsi_hook_enabled']}")
        
        try:
            # Create backtest engine with custom config
            engine = BacktestEngine(
                signal_file_path=str(self.signals_file),
                data_folder_path="backtester/historical_data"
            )
            
            # Apply parameters to config
            profile = engine.config.profiles[0]
            
            # Exit strategy parameters
            profile['exit_strategy']['breakeven_trigger_percent'] = params['breakeven_trigger_percent'] / 100
            profile['exit_strategy']['trail_method'] = params['trail_method']
            profile['exit_strategy']['trail_settings']['pullback_percent'] = params['pullback_percent'] / 100
            profile['exit_strategy']['trail_settings']['atr_period'] = params['atr_period']
            profile['exit_strategy']['trail_settings']['atr_multiplier'] = params['atr_multiplier']
            
            # Momentum exits
            profile['exit_strategy']['momentum_exits']['psar_enabled'] = params['psar_enabled']
            profile['exit_strategy']['momentum_exits']['psar_settings']['start'] = params['psar_start']
            profile['exit_strategy']['momentum_exits']['psar_settings']['increment'] = params['psar_increment']
            profile['exit_strategy']['momentum_exits']['psar_settings']['max'] = params['psar_max']
            
            profile['exit_strategy']['momentum_exits']['rsi_hook_enabled'] = params['rsi_hook_enabled']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['period'] = params['rsi_period']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['overbought_level'] = params['rsi_overbought']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['oversold_level'] = params['rsi_oversold']
            
            # Run simulation
            engine.run_simulation()
            
            # Analyze results
            results_file = Path("backtester/backtest_results.csv")
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
    
    def analyze_results(self, results_file: Path, test_name: str, params: Dict) -> Dict:
        """Analyze backtest results and generate metrics."""
        df = pd.read_csv(results_file)
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        total_pnl = df['pnl'].sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # Calculate max drawdown
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = running_max - cumulative_pnl
        max_drawdown = drawdown.max()
        
        # Sharpe ratio (annualized)
        returns = df['pnl']
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        summary = {
            'test_name': test_name,
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe, 2),
            **params  # Include all parameters
        }
        
        return summary
    
    def generate_report(self):
        """Generate comprehensive optimization report."""
        if not self.results:
            logging.warning("No results to report")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save full results
        full_results_file = self.output_dir / "all_results.csv"
        df.to_csv(full_results_file, index=False)
        
        # Generate summary report
        summary_file = self.output_dir / "optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.results)}\n")
            f.write(f"Mode: {'Quick' if self.quick_mode else 'Full'}\n")
            f.write("="*100 + "\n\n")
            
            # Top 10 by P&L
            f.write("TOP 10 PARAMETER COMBINATIONS (by Total P&L):\n")
            f.write("-"*100 + "\n")
            df_sorted = df.sort_values('total_pnl', ascending=False)
            for i, row in enumerate(df_sorted.head(10).itertuples(), 1):
                f.write(f"\n#{i}. {row.test_name}\n")
                f.write(f"   P&L: ${row.total_pnl:,.2f} | Win Rate: {row.win_rate:.1f}% | PF: {row.profit_factor:.2f}\n")
                f.write(f"   Breakeven: {row.breakeven_trigger_percent}% | Trail: {row.trail_method}\n")
                f.write(f"   PSAR: {row.psar_enabled} | RSI: {row.rsi_hook_enabled}\n")
            
            # Top 10 by Win Rate
            f.write("\n\n" + "="*100 + "\n")
            f.write("TOP 10 PARAMETER COMBINATIONS (by Win Rate):\n")
            f.write("-"*100 + "\n")
            df_sorted = df.sort_values('win_rate', ascending=False)
            for i, row in enumerate(df_sorted.head(10).itertuples(), 1):
                f.write(f"\n#{i}. {row.test_name}\n")
                f.write(f"   Win Rate: {row.win_rate:.1f}% | P&L: ${row.total_pnl:,.2f} | PF: {row.profit_factor:.2f}\n")
                f.write(f"   Breakeven: {row.breakeven_trigger_percent}% | Trail: {row.trail_method}\n")
            
            # Parameter Analysis
            f.write("\n\n" + "="*100 + "\n")
            f.write("PARAMETER IMPACT ANALYSIS:\n")
            f.write("-"*100 + "\n")
            
            for param in ['breakeven_trigger_percent', 'trail_method', 'psar_enabled', 'rsi_hook_enabled']:
                f.write(f"\n{param}:\n")
                grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                for idx, row in grouped.iterrows():
                    f.write(f"  {idx}: Avg P&L ${row['mean']:,.2f} (±${row['std']:,.2f}) | {int(row['count'])} tests\n")
            
            # Recommended Configuration
            f.write("\n\n" + "="*100 + "\n")
            f.write("RECOMMENDED CONFIGURATION:\n")
            f.write("-"*100 + "\n")
            best = df_sorted.iloc[0]
            f.write(f"""
{{
    "exit_strategy": {{
        "breakeven_trigger_percent": {best['breakeven_trigger_percent'] / 100},
        "trail_method": "{best['trail_method']}",
        "trail_settings": {{
            "pullback_percent": {best['pullback_percent'] / 100},
            "atr_period": {best['atr_period']},
            "atr_multiplier": {best['atr_multiplier']}
        }},
        "momentum_exits": {{
            "psar_enabled": {str(best['psar_enabled']).lower()},
            "psar_settings": {{
                "start": {best['psar_start']},
                "increment": {best['psar_increment']},
                "max": {best['psar_max']}
            }},
            "rsi_hook_enabled": {str(best['rsi_hook_enabled']).lower()},
            "rsi_settings": {{
                "period": {best['rsi_period']},
                "overbought_level": {best['rsi_overbought']},
                "oversold_level": {best['rsi_oversold']}
            }}
        }}
    }}
}}
""")
        
        logging.info(f"\n📊 Full results: {full_results_file}")
        logging.info(f"📋 Summary: {summary_file}")
    
    async def run_optimization(self):
        """Main optimization loop."""
        if not self.signals_file.exists():
            logging.error(f"Signals file not found: {self.signals_file}")
            logging.info("Please create: backtester/signals_to_test.txt")
            return
        
        logging.info(f"\n🚀 Starting parameter optimization")
        logging.info(f"Signals: {self.signals_file}")
        logging.info(f"Mode: {'Quick' if self.quick_mode else 'Full'}\n")
        
        # Generate combinations
        combinations = self.generate_combinations()
        total_tests = len(combinations)
        
        # Estimate time
        time_per_test = 5  # seconds (conservative estimate)
        total_time = (total_tests * time_per_test) / 60
        logging.info(f"Estimated time: ~{total_time:.0f} minutes\n")
        
        # Run all tests
        for i, params in enumerate(combinations, 1):
            await self.run_single_test(i, params, total_tests)
        
        # Generate report
        self.generate_report()
        
        logging.info(f"\n✅ Optimization complete! Results in: {self.output_dir}")


async def main():
    """Main entry point with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated parameter optimization for trading strategies"
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick mode (16 tests instead of 3,456)'
    )
    parser.add_argument(
        '--signals', 
        type=str,
        default='backtester/signals_to_test.txt',
        help='Path to signals file'
    )
    parser.add_argument(
        '--params',
        type=str,
        help='Path to custom parameter grid JSON file'
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ParameterOptimizer(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    # Load custom params if provided
    if args.params:
        optimizer.load_custom_grid(args.params)
    
    # Run optimization
    await optimizer.run_optimization()


if __name__ == "__main__":
    asyncio.run(main())
