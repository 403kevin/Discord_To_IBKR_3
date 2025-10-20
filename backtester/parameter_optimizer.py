#!/usr/bin/env python3
"""
parameter_optimizer.py - For regular options (non-0DTE)
Optimizes parameters for scalps and day trades with hold times of hours to days
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
    Parameter optimization for regular options (non-0DTE)
    Tests wider parameter ranges suitable for multi-hour to multi-day holds
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("🚀 ParameterOptimizer initialized")
        logging.info(f"📊 Signals file: {self.signals_file}")
        logging.info(f"⚡ Quick mode: {quick_mode}")
        logging.info(f"📂 Output dir: {self.output_dir}")
        
        # Define parameter grid
        if quick_mode:
            self.param_grid = self.get_quick_grid()
        else:
            self.param_grid = self.get_full_grid()
        
        self.results = []
    
    def get_quick_grid(self):
        """Quick test grid - 16 combinations for scalps/day trades"""
        return {
            'breakeven_trigger_percent': [7, 10],  # Based on 0DTE findings
            'trail_method': ['pullback_percent'],
            'pullback_percent': [8, 10],  # Wider based on test results
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'native_trail_percent': [20, 25],
            'psar_enabled': [True, False],  # Test both for longer holds
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
def get_full_grid(self):
        """Full test grid - FOCUSED but comprehensive (180 combinations vs 2,880)"""
        return {
            'breakeven_trigger_percent': [5, 7, 10, 12, 15],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [8, 10, 12],
            'atr_period': [14],
            'atr_multiplier': [1.5, 2.0],
            'native_trail_percent': [25, 30, 35],
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    async def run_optimization(self):
        """Run parameter optimization"""
        logging.info("\n" + "="*60)
        logging.info("🚀 Starting Parameter Optimization")
        logging.info("="*60)
        
        # Count signals
        signal_count = 0
        with open(self.signals_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and not line.startswith('Trader:'):
                    signal_count += 1
        
        logging.info(f"📊 Found {signal_count} signals to test")
        logging.info(f"⚡ Mode: {'QUICK' if self.quick_mode else 'FULL'}")
        
        # Generate all parameter combinations
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combinations = list(product(*values))
        
        logging.info(f"📊 Generated {len(combinations)} parameter combinations")
        
        # Estimate time
        est_minutes = (len(combinations) * signal_count * 0.5) / 60
        logging.info(f"⏱️ Estimated time: ~{est_minutes:.0f} minutes")
        
        logging.info("\n")
        
        # Run each combination
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            result = await self.run_single_test(idx, params, len(combinations))
            if result:
                self.results.append(result)
        
        # Generate reports
        self.generate_reports()
        
        logging.info(f"\n✅ Optimization complete!")
        logging.info(f"📂 Results saved in: {self.output_dir}")
        
        if self.results:
            best_result = max(self.results, key=lambda x: x['total_pnl'])
            logging.info(f"🏆 Best configuration:")
            logging.info(f"   P&L: ${best_result['total_pnl']:,.2f}")
            logging.info(f"   Win Rate: {best_result['win_rate']:.1f}%")
            logging.info(f"   Profit Factor: {best_result['profit_factor']:.2f}")
    
    async def run_single_test(self, test_num: int, params: Dict, total_tests: int) -> Dict:
        """Run a single backtest with specific parameters"""
        test_name = f"test_{test_num:04d}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Breakeven: {params['breakeven_trigger_percent']}% | "
                    f"Trail: {params['trail_method']} | "
                    f"Native: {params['native_trail_percent']}%")
        
        try:
            # Create backtest engine
            engine = BacktestEngine(
                signal_file_path=str(self.signals_file),
                data_folder_path="backtester/historical_data"
            )
            
            # Run simulation with parameters
            results = engine.run_simulation(params)
            
            if results:
                # Add parameter details to results
                summary = {
                    'test_name': test_name,
                    'total_trades': results['total_trades'],
                    'total_pnl': results['total_pnl'],
                    'win_rate': results['win_rate'],
                    'avg_win': results['avg_win'],
                    'avg_loss': results['avg_loss'],
                    'profit_factor': results['profit_factor'],
                    'max_drawdown': results.get('max_drawdown', 0),
                    'final_capital': results['final_capital'],
                    'return_pct': results['return_pct'],
                    'avg_minutes_held': results.get('avg_minutes_held', 0),
                    **params  # Include all parameters
                }
                
                # Log exit reasons if available
                if 'exit_reasons' in results:
                    summary['exit_reasons'] = results['exit_reasons']
                
                logging.info(f"  ✅ Results: {results['total_trades']} trades | "
                           f"${results['total_pnl']:.0f} P&L | "
                           f"{results['win_rate']:.1f}% WR")
                
                return summary
            else:
                logging.warning(f"  ⚠️ No results returned")
                return None
                
        except Exception as e:
            logging.error(f"  ❌ Error in {test_name}: {str(e)}")
            return None
    
    def generate_reports(self):
        """Generate summary reports"""
        if not self.results:
            logging.warning("No results to report!")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Save all results
        all_results_path = self.output_dir / "all_results.csv"
        df.to_csv(all_results_path, index=False)
        logging.info(f"📊 Saved all results to {all_results_path}")
        
        # Generate summary report
        summary_path = self.output_dir / "optimization_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            # Top 10 configurations
            f.write("TOP 10 PARAMETER COMBINATIONS (by Total P&L):\n")
            f.write("-"*100 + "\n\n")
            
            top_10 = df.nlargest(10, 'total_pnl')
            for idx, row in top_10.iterrows():
                f.write(f"#{idx+1}. {row['test_name']}\n")
                f.write(f"   Win Rate: {row['win_rate']:.1f}% | ")
                f.write(f"P&L: ${row['total_pnl']:.2f} | ")
                f.write(f"PF: {row['profit_factor']:.2f}\n")
                f.write(f"   Breakeven: {row['breakeven_trigger_percent']}% | ")
                f.write(f"Trail: {row['trail_method']}")
                if row['trail_method'] == 'pullback_percent':
                    f.write(f" | Pullback: {row['pullback_percent']}%")
                else:
                    f.write(f" | ATR: {row['atr_period']}/{row['atr_multiplier']}")
                f.write(f" | Native: {row['native_trail_percent']}%\n")
                f.write(f"   Max DD: ${row.get('max_drawdown', 0):.2f} | ")
                f.write(f"Avg Hold: {row.get('avg_minutes_held', 0):.0f} min\n\n")
            
            # Parameter impact analysis
            f.write("\n" + "="*100 + "\n")
            f.write("PARAMETER IMPACT ANALYSIS:\n")
            f.write("-"*100 + "\n\n")
            
            # Analyze each parameter
            param_impact = {}
            for param in ['breakeven_trigger_percent', 'native_trail_percent', 
                         'pullback_percent', 'trail_method', 'psar_enabled', 'rsi_hook_enabled']:
                if param in df.columns:
                    grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                    param_impact[param] = grouped.std().mean() if len(grouped) > 1 else 0
                    
                    f.write(f"{param}:\n")
                    for value, stats in grouped.iterrows():
                        f.write(f"  {value}: Avg P&L ${stats['mean']:.2f} (±${stats['std']:.2f}) | {stats['count']} tests\n")
                    f.write("\n")
            
            # Most impactful parameters
            f.write("MOST IMPACTFUL PARAMETERS (by P&L variance):\n")
            sorted_impact = sorted(param_impact.items(), key=lambda x: x[1], reverse=True)
            for param, impact in sorted_impact[:3]:
                f.write(f"  {param}: Impact score {impact:.2f}\n")
            
            # Recommended configuration
            f.write("\n\n" + "="*100 + "\n")
            f.write("RECOMMENDED CONFIGURATION (Best P&L):\n")
            f.write("-"*100 + "\n\n")
            
            best = df.loc[df['total_pnl'].idxmax()]
            
            f.write("{\n")
            f.write('    "exit_strategy": {\n')
            f.write(f'        "breakeven_trigger_percent": {best["breakeven_trigger_percent"] / 100:.2f},\n')
            f.write(f'        "trail_method": "{best["trail_method"]}",\n')
            f.write(f'        "native_trail_percent": {best["native_trail_percent"] / 100:.2f},\n')
            f.write('        "trail_settings": {\n')
            
            if best['trail_method'] == 'pullback_percent':
                f.write(f'            "pullback_percent": {best["pullback_percent"] / 100:.2f}\n')
            else:
                f.write(f'            "atr_period": {best["atr_period"]},\n')
                f.write(f'            "atr_multiplier": {best["atr_multiplier"]}\n')
            
            f.write('        },\n')
            f.write('        "momentum_exits": {\n')
            f.write(f'            "psar_enabled": {str(best["psar_enabled"]).lower()},\n')
            f.write(f'            "rsi_hook_enabled": {str(best["rsi_hook_enabled"]).lower()}\n')
            f.write('        }\n')
            f.write('    }\n')
            f.write('}\n')
            
            # Alternative best configs
            f.write("\n" + "="*100 + "\n")
            f.write("ALTERNATIVE BEST CONFIGURATIONS:\n")
            f.write("-"*100 + "\n\n")
            
            # Best win rate
            best_wr = df.loc[df['win_rate'].idxmax()]
            f.write(f"Best Win Rate: {best_wr['test_name']} - {best_wr['win_rate']:.1f}% (P&L: ${best_wr['total_pnl']:.2f})\n")
            f.write(f"  Config: BE={best_wr['breakeven_trigger_percent']}%, "
                   f"Native={best_wr['native_trail_percent']}%, {best_wr['trail_method']}\n\n")
            
            # Best profit factor
            best_pf = df.loc[df['profit_factor'].idxmax()]
            f.write(f"Best Profit Factor: {best_pf['test_name']} - PF={best_pf['profit_factor']:.2f} (P&L: ${best_pf['total_pnl']:.2f})\n")
            f.write(f"  Config: BE={best_pf['breakeven_trigger_percent']}%, "
                   f"Native={best_pf['native_trail_percent']}%, {best_pf['trail_method']}\n\n")
            
            # Lowest drawdown
            if 'max_drawdown' in df.columns:
                best_dd = df.loc[df['max_drawdown'].idxmax()]
                f.write(f"Lowest Drawdown: {best_dd['test_name']} - DD=${best_dd['max_drawdown']:.2f} (P&L: ${best_dd['total_pnl']:.2f})\n")
                f.write(f"  Config: BE={best_dd['breakeven_trigger_percent']}%, "
                       f"Native={best_dd['native_trail_percent']}%, {best_dd['trail_method']}\n")
        
        logging.info(f"📋 Saved summary report to {summary_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parameter optimization for options trading strategies"
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick mode (fewer tests, ~16 combinations)'
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
        with open(args.params, 'r') as f:
            optimizer.param_grid = json.load(f)
        logging.info(f"Loaded custom parameter grid from {args.params}")
    
    # Run optimization
    await optimizer.run_optimization()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)
