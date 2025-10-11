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
            await engine.run_simulation()
            
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
        
        if df.empty:
            return {
                'test_name': test_name,
                'total_trades': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                **params
            }
        
        # Calculate metrics
        total_trades = len(df)
        total_pnl = df['pnl'].sum()
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else total_wins
        
        # Max drawdown
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
        
        # Sharpe ratio (simplified)
        returns = df['pnl']
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        return {
            'test_name': test_name,
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_win': winning_trades['pnl'].mean() if not winning_trades.empty else 0,
            'avg_loss': losing_trades['pnl'].mean() if not losing_trades.empty else 0,
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min(),
            **params
        }
    
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
            for i, row in df_sorted.head(10).iterrows():
                f.write(f"\n#{i+1}. {row['test_name']} - P&L: ${row['total_pnl']:.2f}\n")
                f.write(f"   Trades: {row['total_trades']} | Win Rate: {row['win_rate']:.1f}% | ")
                f.write(f"Profit Factor: {row['profit_factor']:.2f}\n")
                f.write(f"   Breakeven: {row['breakeven_trigger_percent']}% | Trail: {row['trail_method']}\n")
                if row['trail_method'] == 'atr':
                    f.write(f"   ATR: {row['atr_multiplier']}x | ")
                else:
                    f.write(f"   Pullback: {row['pullback_percent']}% | ")
                f.write(f"PSAR: {row['psar_enabled']} | RSI Hook: {row['rsi_hook_enabled']}\n")
            
            # Parameter impact analysis
            f.write("\n" + "="*100 + "\n")
            f.write("PARAMETER ANALYSIS:\n")
            f.write("-"*100 + "\n")
            
            # Analyze each parameter's impact
            for param in ['breakeven_trigger_percent', 'trail_method', 'psar_enabled', 'rsi_hook_enabled']:
                f.write(f"\n{param.replace('_', ' ').title()}:\n")
                grouped = df.groupby(param)['total_pnl'].mean().sort_values(ascending=False)
                for value, avg_pnl in grouped.items():
                    f.write(f"  {value}: Avg P&L ${avg_pnl:.2f}\n")
            
            # Best overall configuration
            f.write("\n" + "="*100 + "\n")
            f.write("RECOMMENDED CONFIGURATION:\n")
            f.write("-"*100 + "\n")
            best = df_sorted.iloc[0]
            f.write(f"""
"exit_strategy": {{
    "breakeven_trigger_percent": {best['breakeven_trigger_percent']/100:.2f},
    "trail_method": "{best['trail_method']}",
    "trail_settings": {{
        "pullback_percent": {best['pullback_percent']/100:.2f},
        "atr_period": {int(best['atr_period'])},
        "atr_multiplier": {best['atr_multiplier']}
    }},
    "momentum_exits": {{
        "psar_enabled": {str(best['psar_enabled']).lower()},
        "psar_settings": {{"start": {best['psar_start']}, "increment": {best['psar_increment']}, "max": {best['psar_max']}}},
        "rsi_hook_enabled": {str(best['rsi_hook_enabled']).lower()},
        "rsi_settings": {{"period": {int(best['rsi_period'])}, "overbought_level": {int(best['rsi_overbought'])}, "oversold_level": {int(best['rsi_oversold'])}}}
    }}
}}
""")
        
        # Print to console
        print("\n" + open(summary_file).read())
        print(f"\n✅ Full results saved to: {full_results_file}")
        print(f"✅ Summary saved to: {summary_file}\n")
    
    async def run_optimization(self):
        """Execute the complete optimization workflow."""
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
