#!/usr/bin/env python3
"""
parameter_optimizer.py - COMPLETE VERSION WITH NATIVE TRAIL
Includes native_trail_percent in parameter optimization
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
    COMPLETE VERSION: Includes native trailing stop in optimization
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("🚀 ParameterOptimizer initialized (COMPLETE VERSION with native trail)")
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
        """Quick test grid - includes native trail"""
        return {
            'breakeven_trigger_percent': [10, 15],
            'trail_method': ['pullback_percent'],  # Simplified for quick test
            'pullback_percent': [10],
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'native_trail_percent': [20, 30],  # Native trailing stop: 20% or 30%
            'psar_enabled': [True, False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],  # Simplified
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def get_full_grid(self):
        """Full parameter grid - comprehensive testing with native trail"""
        return {
            'breakeven_trigger_percent': [5, 10, 15, 20],
            'trail_method': ['atr', 'pullback_percent'],
            'pullback_percent': [8, 10, 12, 15],
            'atr_period': [10, 14, 20],
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
            'native_trail_percent': [15, 20, 25, 30, 35],  # Native trail: 15-35%
            'psar_enabled': [True, False],
            'psar_start': [0.01, 0.02, 0.03],
            'psar_increment': [0.01, 0.02, 0.03],
            'psar_max': [0.1, 0.2, 0.3],
            'rsi_hook_enabled': [True, False],
            'rsi_period': [10, 14, 20],
            'rsi_overbought': [65, 70, 75],
            'rsi_oversold': [25, 30, 35]
        }
    
    def generate_combinations(self):
        """Generate all parameter combinations"""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        
        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        logging.info(f"📊 Generated {len(combinations)} parameter combinations")
        
        return combinations
    
    async def run_single_test(self, test_num: int, params: Dict, total_tests: int) -> Dict:
        """Run a single backtest with specific parameters"""
        test_name = f"test_{test_num:04d}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Breakeven: {params['breakeven_trigger_percent']}% | Trail: {params['trail_method']}")
        logging.info(f"  Native Trail: {params['native_trail_percent']}% | Pullback: {params['pullback_percent']}%")
        logging.info(f"  PSAR: {params['psar_enabled']} | RSI: {params['rsi_hook_enabled']}")
        
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
                if 'exit_reasons' in results and results['exit_reasons']:
                    logging.debug(f"  Exit reasons: {results['exit_reasons']}")
                
                self.results.append(summary)
                logging.info(f"✅ Test {test_num} complete: P&L=${summary['total_pnl']:.2f}, WR={summary['win_rate']:.1f}%, PF={summary['profit_factor']:.2f}")
                return summary
            else:
                logging.warning(f"⚠️ No results returned for {test_name}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Test {test_name} failed: {e}", exc_info=True)
            return None
    
    def generate_report(self):
        """Generate comprehensive optimization report"""
        if not self.results:
            logging.warning("No results to report")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save all results
        all_results_file = self.output_dir / "all_results.csv"
        df.to_csv(all_results_file, index=False)
        logging.info(f"📊 Full results saved to {all_results_file}")
        
        # Generate summary report
        summary_file = self.output_dir / "optimization_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PARAMETER OPTIMIZATION SUMMARY (WITH NATIVE TRAILING STOP)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            # Top 10 by P&L
            f.write("TOP 10 PARAMETER COMBINATIONS (by Total P&L):\n")
            f.write("-"*100 + "\n\n")
            
            top_10 = df.nlargest(10, 'total_pnl')
            for i, row in enumerate(top_10.itertuples(), 1):
                f.write(f"#{i}. {row.test_name}\n")
                f.write(f"   Win Rate: {row.win_rate:.1f}% | P&L: ${row.total_pnl:,.2f} | PF: {row.profit_factor:.2f}\n")
                f.write(f"   Breakeven: {row.breakeven_trigger_percent}% | Trail: {row.trail_method} | Native: {row.native_trail_percent}%\n")
                f.write(f"   Pullback: {row.pullback_percent}% | PSAR: {row.psar_enabled} | RSI: {row.rsi_hook_enabled}\n")
                f.write(f"   Max DD: ${row.max_drawdown:.2f} | Avg Hold: {row.avg_minutes_held:.0f} min\n\n")
            
            # Parameter impact analysis
            f.write("\n" + "="*100 + "\n")
            f.write("PARAMETER IMPACT ANALYSIS:\n")
            f.write("-"*100 + "\n\n")
            
            # Analyze each parameter
            for param in ['breakeven_trigger_percent', 'trail_method', 'native_trail_percent', 
                         'pullback_percent', 'psar_enabled', 'rsi_hook_enabled']:
                f.write(f"{param}:\n")
                grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                for value, stats in grouped.iterrows():
                    f.write(f"  {value}: Avg P&L ${stats['mean']:,.2f} (±${stats['std']:,.2f}) | {int(stats['count'])} tests\n")
                f.write("\n")
            
            # Find most impactful parameter
            f.write("MOST IMPACTFUL PARAMETERS (by P&L variance):\n")
            impact_scores = {}
            for param in ['breakeven_trigger_percent', 'native_trail_percent', 'pullback_percent']:
                grouped = df.groupby(param)['total_pnl'].mean()
                impact_scores[param] = grouped.std()
            
            sorted_impact = sorted(impact_scores.items(), key=lambda x: x[1], reverse=True)
            for param, impact in sorted_impact[:3]:
                f.write(f"  {param}: Impact score {impact:.2f}\n")
            f.write("\n")
            
            # Recommended configuration
            best = df.loc[df['total_pnl'].idxmax()]
            f.write("\n" + "="*100 + "\n")
            f.write("RECOMMENDED CONFIGURATION (Best P&L):\n")
            f.write("-"*100 + "\n\n")
            f.write("{\n")
            f.write('    "exit_strategy": {\n')
            f.write(f'        "breakeven_trigger_percent": {best.breakeven_trigger_percent/100:.2f},\n')
            f.write(f'        "trail_method": "{best.trail_method}",\n')
            f.write(f'        "native_trail_percent": {best.native_trail_percent/100:.2f},  # Native trailing stop\n')
            f.write('        "trail_settings": {\n')
            f.write(f'            "pullback_percent": {best.pullback_percent/100:.2f},\n')
            f.write(f'            "atr_period": {int(best.atr_period)},\n')
            f.write(f'            "atr_multiplier": {best.atr_multiplier}\n')
            f.write('        },\n')
            f.write('        "momentum_exits": {\n')
            f.write(f'            "psar_enabled": {str(best.psar_enabled).lower()},\n')
            f.write('            "psar_settings": {\n')
            f.write(f'                "start": {best.psar_start},\n')
            f.write(f'                "increment": {best.psar_increment},\n')
            f.write(f'                "max": {best.psar_max}\n')
            f.write('            },\n')
            f.write(f'            "rsi_hook_enabled": {str(best.rsi_hook_enabled).lower()},\n')
            f.write('            "rsi_settings": {\n')
            f.write(f'                "period": {int(best.rsi_period)},\n')
            f.write(f'                "overbought_level": {int(best.rsi_overbought)},\n')
            f.write(f'                "oversold_level": {int(best.rsi_oversold)}\n')
            f.write('            }\n')
            f.write('        }\n')
            f.write('    }\n')
            f.write('}\n')
            
            # Best by other metrics
            f.write("\n" + "="*100 + "\n")
            f.write("ALTERNATIVE BEST CONFIGURATIONS:\n")
            f.write("-"*100 + "\n\n")
            
            # Best win rate
            best_wr = df.loc[df['win_rate'].idxmax()]
            f.write(f"Best Win Rate: {best_wr.test_name} - {best_wr.win_rate:.1f}% (P&L: ${best_wr.total_pnl:.2f})\n")
            f.write(f"  Config: BE={best_wr.breakeven_trigger_percent}%, Native={best_wr.native_trail_percent}%, {best_wr.trail_method}\n\n")
            
            # Best profit factor
            valid_pf = df[df['profit_factor'] < float('inf')]
            if not valid_pf.empty:
                best_pf = valid_pf.loc[valid_pf['profit_factor'].idxmax()]
                f.write(f"Best Profit Factor: {best_pf.test_name} - PF={best_pf.profit_factor:.2f} (P&L: ${best_pf.total_pnl:.2f})\n")
                f.write(f"  Config: BE={best_pf.breakeven_trigger_percent}%, Native={best_pf.native_trail_percent}%, {best_pf.trail_method}\n\n")
            
            # Lowest drawdown
            if 'max_drawdown' in df.columns:
                best_dd = df.loc[df['max_drawdown'].idxmax()]  # Least negative
                f.write(f"Lowest Drawdown: {best_dd.test_name} - DD=${best_dd.max_drawdown:.2f} (P&L: ${best_dd.total_pnl:.2f})\n")
                f.write(f"  Config: BE={best_dd.breakeven_trigger_percent}%, Native={best_dd.native_trail_percent}%, {best_dd.trail_method}\n")
        
        logging.info(f"📊 Summary saved to {summary_file}")
    
    async def run_optimization(self):
        """Main optimization loop"""
        logging.info("\n" + "="*60)
        logging.info("🚀 Starting Parameter Optimization (WITH NATIVE TRAILING STOP)")
        logging.info("="*60)
        
        # Check signals file
        if not self.signals_file.exists():
            logging.error(f"❌ Signals file not found: {self.signals_file}")
            return
        
        # Count signals
        with open(self.signals_file, 'r') as f:
            signal_lines = [line for line in f if line.strip() and not line.startswith('#')]
        
        logging.info(f"📊 Found {len(signal_lines)} signals to test")
        logging.info(f"⚡ Mode: {'QUICK' if self.quick_mode else 'FULL'}")
        
        # Generate combinations
        combinations = self.generate_combinations()
        
        if not combinations:
            logging.error("❌ No parameter combinations generated!")
            return
        
        total_tests = len(combinations)
        logging.info(f"📊 Testing {total_tests} parameter combinations")
        logging.info(f"🎯 Including native trailing stop testing (15-35%)")
        
        # Estimate time
        time_per_test = 5  # seconds
        total_time = (total_tests * time_per_test) / 60
        logging.info(f"⏱️ Estimated time: ~{total_time:.0f} minutes\n")
        
        # Run tests
        for i, params in enumerate(combinations, 1):
            result = await self.run_single_test(i, params, total_tests)
            
            # Progress update every 10 tests
            if i % 10 == 0:
                logging.info(f"\n📊 PROGRESS: {i}/{total_tests} tests complete ({i/total_tests*100:.1f}%)")
                if self.results:
                    best_so_far = max(self.results, key=lambda x: x['total_pnl'])
                    logging.info(f"   Best P&L so far: ${best_so_far['total_pnl']:,.2f}")
                    logging.info(f"   Best config: BE={best_so_far['breakeven_trigger_percent']}%, Native={best_so_far['native_trail_percent']}%")
        
        # Generate report
        self.generate_report()
        
        logging.info(f"\n✅ Optimization complete!")
        logging.info(f"📂 Results saved in: {self.output_dir}")
        
        if self.results:
            best_result = max(self.results, key=lambda x: x['total_pnl'])
            logging.info(f"🏆 Best configuration:")
            logging.info(f"   P&L: ${best_result['total_pnl']:,.2f}")
            logging.info(f"   Win Rate: {best_result['win_rate']:.1f}%")
            logging.info(f"   Native Trail: {best_result['native_trail_percent']}%")
            logging.info(f"   Breakeven: {best_result['breakeven_trigger_percent']}%")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parameter optimization for options backtesting (with native trailing stop)"
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick mode (fewer tests)'
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
