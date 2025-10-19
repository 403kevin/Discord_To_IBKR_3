#!/usr/bin/env python3
"""
parameter_optimizer_0DTE.py - SPECIALIZED FOR 0DTE OPTIONS
For ultra-short-term 0DTE options with typical hold times of 15-60 minutes
Uses much tighter parameters optimized for rapid price action
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


class ParameterOptimizer0DTE:
    """
    SPECIALIZED FOR 0DTE OPTIONS
    Tight parameters for rapid scalping (15-60 minute holds)
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/0DTE_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("🚀 ParameterOptimizer initialized (0DTE SPECIALIZED)")
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
    """Quick test grid for 0DTE - Based on proven winners"""
    return {
        'breakeven_trigger_percent': [7, 10],  # 7% was best
        'trail_method': ['pullback_percent'],
        'pullback_percent': [10, 12],  # 10% was best
        'atr_period': [14],
        'atr_multiplier': [1.5],
        'native_trail_percent': [20, 25],  # 20% was best
        'psar_enabled': [False],
        'psar_start': [0.02],
        'psar_increment': [0.02],
        'psar_max': [0.2],
        'rsi_hook_enabled': [False],
        'rsi_period': [14],
        'rsi_overbought': [70],
        'rsi_oversold': [30]
    }

def get_full_grid(self):
    """Full test grid for 0DTE - REVISED based on test results"""
    return {
        'breakeven_trigger_percent': [5, 7, 10, 12, 15],  # Focus on 7%+ which worked
        'trail_method': ['pullback_percent'],
        'pullback_percent': [8, 10, 12, 15, 20],  # 10% was best, test wider
        'atr_period': [14],
        'atr_multiplier': [1.5],
        'native_trail_percent': [15, 20, 25, 30],  # 20% was optimal
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
        logging.info("🚀 Starting 0DTE Parameter Optimization")
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
        logging.info(f"🎯 Optimized for 0DTE rapid scalping (15-60 min holds)")
        
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
        
        logging.info(f"\n✅ 0DTE Optimization complete!")
        logging.info(f"📂 Results saved in: {self.output_dir}")
        
        if self.results:
            best_result = max(self.results, key=lambda x: x['total_pnl'])
            logging.info(f"🏆 Best 0DTE configuration:")
            logging.info(f"   P&L: ${best_result['total_pnl']:,.2f}")
            logging.info(f"   Win Rate: {best_result['win_rate']:.1f}%")
            logging.info(f"   Native Trail: {best_result['native_trail_percent']}%")
            logging.info(f"   Breakeven: {best_result['breakeven_trigger_percent']}%")
            logging.info(f"   Avg Hold: {best_result['avg_minutes_held']:.0f} min")
    
    async def run_single_test(self, test_num: int, params: Dict, total_tests: int) -> Dict:
        """Run a single backtest with specific parameters"""
        test_name = f"test_{test_num:04d}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Breakeven: {params['breakeven_trigger_percent']}% | Pullback: {params['pullback_percent']}%")
        logging.info(f"  Native Trail: {params['native_trail_percent']}%")
        
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
                           f"{results['win_rate']:.1f}% WR | "
                           f"{results.get('avg_minutes_held', 0):.0f}min hold")
                
                return summary
            else:
                logging.warning(f"  ⚠️ No results returned")
                return None
                
        except Exception as e:
            logging.error(f"❌ Test {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_reports(self):
        """Generate optimization summary reports"""
        if not self.results:
            logging.warning("No results to report")
            return
        
        # Save all results to CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"📊 Saved detailed results to {csv_path}")
        
        # Generate summary report
        self.generate_summary_report(df)
    
    def generate_summary_report(self, df):
        """Generate human-readable summary"""
        summary_path = self.output_dir / "optimization_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("0DTE PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Optimized for ultra-short-term 0DTE scalping (15-60 minute holds)\n")
            f.write("="*100 + "\n\n")
            
            # Top 10 by P&L
            f.write("TOP 10 PARAMETER COMBINATIONS (by Total P&L):\n")
            f.write("-"*100 + "\n\n")
            
            top_10_pnl = df.nlargest(10, 'total_pnl')
            for idx, row in enumerate(top_10_pnl.itertuples(), 1):
                f.write(f"#{idx}. {row.test_name}\n")
                f.write(f"   Win Rate: {row.win_rate:.1f}% | P&L: ${row.total_pnl:.2f} | PF: {row.profit_factor:.2f}\n")
                f.write(f"   Breakeven: {row.breakeven_trigger_percent}% | Pullback: {row.pullback_percent}% | Native: {row.native_trail_percent}%\n")
                f.write(f"   Max DD: ${row.max_drawdown:.2f} | Avg Hold: {row.avg_minutes_held:.0f} min\n\n")
            
            # Parameter impact analysis
            f.write("\n" + "="*100 + "\n")
            f.write("PARAMETER IMPACT ANALYSIS:\n")
            f.write("-"*100 + "\n\n")
            
            for param in ['breakeven_trigger_percent', 'native_trail_percent', 'pullback_percent']:
                if param in df.columns:
                    f.write(f"{param}:\n")
                    grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                    for value, stats in grouped.iterrows():
                        f.write(f"  {value}%: Avg P&L ${stats['mean']:.2f} (±${stats['std']:.2f}) | {stats['count']:.0f} tests\n")
                    f.write("\n")
            
            # Most impactful parameters
            f.write("MOST IMPACTFUL PARAMETERS (by P&L variance):\n")
            impact_scores = {}
            for param in ['breakeven_trigger_percent', 'native_trail_percent', 'pullback_percent']:
                if param in df.columns:
                    impact_scores[param] = df.groupby(param)['total_pnl'].std().mean()
            
            for param, score in sorted(impact_scores.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {param}: Impact score {score:.2f}\n")
            
            # Recommended configuration
            f.write("\n\n" + "="*100 + "\n")
            f.write("RECOMMENDED 0DTE CONFIGURATION (Best P&L):\n")
            f.write("-"*100 + "\n\n")
            
            best = df.loc[df['total_pnl'].idxmax()]
            
            f.write("{\n")
            f.write('    "exit_strategy": {\n')
            f.write(f'        "breakeven_trigger_percent": {best["breakeven_trigger_percent"]/100:.3f},  # {best["breakeven_trigger_percent"]}%\n')
            f.write(f'        "trail_method": "pullback_percent",  # Best for 0DTE\n')
            f.write(f'        "native_trail_percent": {best["native_trail_percent"]/100:.2f},  # {best["native_trail_percent"]}% safety net\n')
            f.write('        "trail_settings": {\n')
            f.write(f'            "pullback_percent": {best["pullback_percent"]/100:.3f}  # {best["pullback_percent"]}% pullback\n')
            f.write('        },\n')
            f.write('        "momentum_exits": {\n')
            f.write('            "psar_enabled": false,  # Not useful for 0DTE\n')
            f.write('            "rsi_hook_enabled": false  # Not useful for 0DTE\n')
            f.write('        }\n')
            f.write('    }\n')
            f.write('}\n')
            
            f.write("\n💡 KEY INSIGHTS FOR 0DTE TRADING:\n")
            f.write("-"*100 + "\n")
            f.write(f"• Average hold time: {df['avg_minutes_held'].mean():.0f} minutes\n")
            f.write(f"• Best win rate: {df['win_rate'].max():.1f}%\n")
            f.write(f"• Best profit factor: {df['profit_factor'].max():.2f}\n")
            f.write(f"• Lowest drawdown: ${df['max_drawdown'].min():.2f}\n")
            f.write("\n")
            
            # Alternative best configs
            f.write("\n" + "="*100 + "\n")
            f.write("ALTERNATIVE BEST CONFIGURATIONS:\n")
            f.write("-"*100 + "\n\n")
            
            best_wr = df.loc[df['win_rate'].idxmax()]
            f.write(f"Best Win Rate: {best_wr['test_name']} - {best_wr['win_rate']:.1f}% (P&L: ${best_wr['total_pnl']:.2f})\n")
            f.write(f"  Config: BE={best_wr['breakeven_trigger_percent']}%, Pullback={best_wr['pullback_percent']}%, Native={best_wr['native_trail_percent']}%\n\n")
            
            best_pf = df.loc[df['profit_factor'].idxmax()]
            f.write(f"Best Profit Factor: {best_pf['test_name']} - PF={best_pf['profit_factor']:.2f} (P&L: ${best_pf['total_pnl']:.2f})\n")
            f.write(f"  Config: BE={best_pf['breakeven_trigger_percent']}%, Pullback={best_pf['pullback_percent']}%, Native={best_pf['native_trail_percent']}%\n\n")
            
            best_dd = df.loc[df['max_drawdown'].idxmin()]
            f.write(f"Lowest Drawdown: {best_dd['test_name']} - DD=${best_dd['max_drawdown']:.2f} (P&L: ${best_dd['total_pnl']:.2f})\n")
            f.write(f"  Config: BE={best_dd['breakeven_trigger_percent']}%, Pullback={best_dd['pullback_percent']}%, Native={best_dd['native_trail_percent']}%\n")
        
        logging.info(f"📋 Saved 0DTE summary report to {summary_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Parameter optimization for 0DTE options (ultra-short-term scalping)"
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick mode (fewer tests, ~8 combinations)'
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
    optimizer = ParameterOptimizer0DTE(
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
