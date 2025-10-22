#!/usr/bin/env python3
"""
parameter_optimizer.py - CONSOLIDATED VERSION
- One optimizer for all day trading (0DTE + regular)
- No PSAR/RSI (useless for scalping)
- Expanded ATR parameters
- Native-trail-only test section in full mode
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtester.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ParameterOptimizer:
    """
    Consolidated parameter optimizer for day trading
    Tests dynamic exits (breakeven, ATR, pullback) + native trail
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.results = []
        self.native_trail_only_results = []
        
        logging.info("="*80)
        logging.info("CONSOLIDATED PARAMETER OPTIMIZER")
        logging.info("="*80)
        logging.info(f"Signals file: {self.signals_file}")
        logging.info(f"Quick mode: {quick_mode}")
        logging.info(f"Output dir: {self.output_dir}")
        
        # Define parameter grids
        if quick_mode:
            self.param_grid = self.get_quick_grid()
        else:
            self.param_grid = self.get_full_grid()
            self.native_trail_only_grid = self.get_native_trail_only_grid()
    
    def get_quick_grid(self) -> Dict:
        """Quick test parameters - streamlined for speed"""
        return {
            'breakeven_trigger_percent': [7, 10, 12],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [10, 12],
            'atr_period': [14],
            'atr_multiplier': [1.5, 2.0],
            'native_trail_percent': [25, 30]
        }
    
    def get_full_grid(self) -> Dict:
        """Full test parameters - comprehensive testing"""
        return {
            'breakeven_trigger_percent': [5, 7, 10, 12, 15],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [8, 10, 12, 15],
            'atr_period': [5, 10, 14, 20, 30],
            'atr_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            'native_trail_percent': [25, 30, 35]
        }
    
    def get_native_trail_only_grid(self) -> Dict:
        """
        Native trail ONLY test - for curiosity
        Disables all other exits (breakeven/pullback never trigger)
        """
        return {
            'breakeven_trigger_percent': [100],  # Never triggers
            'trail_method': ['pullback_percent'],
            'pullback_percent': [100],  # Never triggers
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'native_trail_percent': [10, 12, 15, 17, 20, 25, 30]  # User's curiosity range
        }
    
    async def run_optimization(self):
        """Run optimization"""
        logging.info("\n" + "="*80)
        logging.info("STARTING OPTIMIZATION")
        logging.info("="*80)
        
        # Generate combinations
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        combinations = list(product(*values))
        
        total_tests = len(combinations)
        
        # Add native-trail-only tests in full mode
        if not self.quick_mode:
            nt_keys = list(self.native_trail_only_grid.keys())
            nt_values = [self.native_trail_only_grid[k] for k in nt_keys]
            nt_combinations = list(product(*nt_values))
            total_tests += len(nt_combinations)
            
            logging.info(f"Dynamic exit tests: {len(combinations)}")
            logging.info(f"Native-trail-only tests: {len(nt_combinations)}")
        
        logging.info(f"Total tests: {total_tests}")
        
        est_minutes = (total_tests * 0.5) / 60
        logging.info(f"Estimated time: ~{est_minutes:.0f} minutes\n")
        
        # Run dynamic exit tests
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            result = await self.run_single_test(idx, params, total_tests, test_type='dynamic')
            if result:
                self.results.append(result)
        
        # Run native-trail-only tests (full mode only)
        if not self.quick_mode:
            logging.info("\n" + "="*80)
            logging.info("NATIVE TRAIL ONLY TESTS (Curiosity)")
            logging.info("="*80 + "\n")
            
            for idx, combo in enumerate(nt_combinations, len(combinations) + 1):
                params = dict(zip(nt_keys, combo))
                result = await self.run_single_test(idx, params, total_tests, test_type='native_only')
                if result:
                    self.native_trail_only_results.append(result)
        
        # Generate reports
        self.generate_reports()
        
        logging.info("\n" + "="*80)
        logging.info("OPTIMIZATION COMPLETE")
        logging.info("="*80)
        logging.info(f"Results saved in: {self.output_dir}")
        
        if self.results:
            best_result = max(self.results, key=lambda x: x['total_pnl'])
            logging.info(f"\nBest dynamic exit config:")
            logging.info(f"   P&L: ${best_result['total_pnl']:,.2f}")
            logging.info(f"   Win Rate: {best_result['win_rate']:.1f}%")
            logging.info(f"   Profit Factor: {best_result['profit_factor']:.2f}")
        
        if self.native_trail_only_results:
            best_nt = max(self.native_trail_only_results, key=lambda x: x['total_pnl'])
            logging.info(f"\nBest native-trail-only config:")
            logging.info(f"   P&L: ${best_nt['total_pnl']:,.2f}")
            logging.info(f"   Native Trail: {best_nt['native_trail_percent']}%")
    
    async def run_single_test(self, test_num: int, params: Dict, total_tests: int, test_type: str = 'dynamic') -> Dict:
        """Run a single backtest"""
        test_name = f"test_{test_num:04d}_{test_type}"
        
        if test_type == 'native_only':
            logging.info(f"[{test_num}/{total_tests}] {test_name} | Native Trail: {params['native_trail_percent']}%")
        else:
            logging.info(f"[{test_num}/{total_tests}] {test_name}")
            logging.info(f"  BE: {params['breakeven_trigger_percent']}% | "
                        f"Trail: {params['trail_method']} | "
                        f"Native: {params['native_trail_percent']}%")
        
        try:
            engine = BacktestEngine(
                signal_file_path=str(self.signals_file),
                data_folder_path="backtester/historical_data"
            )
            
            results = engine.run_simulation(params)
            
            if results:
                summary = {
                    'test_name': test_name,
                    'test_type': test_type,
                    'total_trades': results['total_trades'],
                    'total_pnl': results['total_pnl'],
                    'win_rate': results['win_rate'],
                    'avg_win': results['avg_win'],
                    'avg_loss': results['avg_loss'],
                    'profit_factor': results['profit_factor'],
                    'final_capital': results['final_capital'],
                    'return_pct': results['return_pct'],
                    'avg_minutes_held': results.get('avg_minutes_held', 0),
                    **params
                }
                
                if 'exit_reasons' in results:
                    summary['exit_reasons'] = results['exit_reasons']
                
                logging.info(f"  Results: {results['total_trades']} trades | "
                           f"${results['total_pnl']:.0f} P&L | "
                           f"{results['win_rate']:.1f}% WR | "
                           f"{results.get('avg_minutes_held', 0):.0f}min hold")
                
                return summary
            else:
                logging.warning(f"  No results returned")
                return None
                
        except Exception as e:
            logging.error(f"  Error in {test_name}: {str(e)}")
            return None
    
    def generate_reports(self):
        """Generate comprehensive summary reports"""
        if not self.results and not self.native_trail_only_results:
            logging.warning("No results to report!")
            return
        
        # Save all results to CSV
        all_results = self.results + self.native_trail_only_results
        df = pd.DataFrame(all_results)
        csv_path = self.output_dir / "all_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved all results: {csv_path}")
        
        # Generate summary report
        summary_path = self.output_dir / "optimization_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("CONSOLIDATED PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # DYNAMIC EXIT RESULTS
            if self.results:
                df_dynamic = pd.DataFrame(self.results)
                
                f.write("=" * 100 + "\n")
                f.write("DYNAMIC EXIT STRATEGIES (Breakeven + ATR/Pullback + Native Trail)\n")
                f.write("-" * 100 + "\n\n")
                
                f.write(f"Tests run: {len(self.results)}\n")
                f.write(f"Profitable configs: {sum(1 for r in self.results if r['total_pnl'] > 0)}\n\n")
                
                # Top 10 dynamic configs
                f.write("TOP 10 DYNAMIC EXIT CONFIGURATIONS:\n")
                f.write("-" * 100 + "\n\n")
                
                top_10 = df_dynamic.nlargest(10, 'total_pnl')
                for idx, row in top_10.iterrows():
                    f.write(f"#{idx+1}. {row['test_name']}\n")
                    f.write(f"   Win Rate: {row['win_rate']:.1f}% | ")
                    f.write(f"P&L: ${row['total_pnl']:.2f} | ")
                    f.write(f"PF: {row['profit_factor']:.2f}\n")
                    f.write(f"   Breakeven: {row['breakeven_trigger_percent']}% | ")
                    f.write(f"Trail: {row['trail_method']}")
                    if row['trail_method'] == 'pullback_percent':
                        f.write(f" {row['pullback_percent']}% | ")
                    else:
                        f.write(f" ATR {row['atr_period']}p×{row['atr_multiplier']} | ")
                    f.write(f"Native: {row['native_trail_percent']}%\n")
                    f.write(f"   Avg Hold: {row.get('avg_minutes_held', 0):.0f} min\n")
                    if 'exit_reasons' in row and row['exit_reasons']:
                        f.write(f"   Exits: {row['exit_reasons']}\n")
                    f.write("\n")
                
                # Parameter impact
                f.write("\n" + "=" * 100 + "\n")
                f.write("PARAMETER IMPACT ANALYSIS:\n")
                f.write("-" * 100 + "\n\n")
                
                for param in ['breakeven_trigger_percent', 'trail_method', 'native_trail_percent']:
                    if param in df_dynamic.columns:
                        grouped = df_dynamic.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                        f.write(f"{param}:\n")
                        for value, stats in grouped.iterrows():
                            f.write(f"  {value}: Avg P&L ${stats['mean']:.2f} (±${stats['std']:.2f}) | {int(stats['count'])} tests\n")
                        f.write("\n")
            
            # NATIVE TRAIL ONLY RESULTS
            if self.native_trail_only_results:
                df_native = pd.DataFrame(self.native_trail_only_results)
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("NATIVE TRAIL ONLY TESTS (No Breakeven, No Dynamic Trail)\n")
                f.write("-" * 100 + "\n\n")
                
                f.write(f"Tests run: {len(self.native_trail_only_results)}\n")
                f.write(f"Profitable configs: {sum(1 for r in self.native_trail_only_results if r['total_pnl'] > 0)}\n\n")
                
                # All native trail results (sorted)
                f.write("ALL NATIVE TRAIL ONLY RESULTS (sorted by P&L):\n")
                f.write("-" * 100 + "\n\n")
                
                sorted_native = df_native.sort_values('total_pnl', ascending=False)
                for idx, row in sorted_native.iterrows():
                    f.write(f"Native Trail: {row['native_trail_percent']}% | ")
                    f.write(f"P&L: ${row['total_pnl']:.2f} | ")
                    f.write(f"Win Rate: {row['win_rate']:.1f}% | ")
                    f.write(f"PF: {row['profit_factor']:.2f} | ")
                    f.write(f"Hold: {row.get('avg_minutes_held', 0):.0f}min\n")
                    if 'exit_reasons' in row and row['exit_reasons']:
                        f.write(f"   Exits: {row['exit_reasons']}\n")
                
                f.write("\n")
                
                # Native trail impact
                f.write("NATIVE TRAIL IMPACT:\n")
                f.write("-" * 100 + "\n")
                grouped = df_native.groupby('native_trail_percent')['total_pnl'].agg(['mean', 'std'])
                f.write(f"{'Trail %':<10} {'Avg P&L':<15} {'Std Dev':<15}\n")
                f.write("-" * 40 + "\n")
                for trail_pct, stats in grouped.iterrows():
                    f.write(f"{trail_pct:<10} ${stats['mean']:<14.2f} ±${stats['std']:<14.2f}\n")
            
            # COMPARISON
            if self.results and self.native_trail_only_results:
                f.write("\n\n" + "=" * 100 + "\n")
                f.write("DYNAMIC VS NATIVE-ONLY COMPARISON:\n")
                f.write("-" * 100 + "\n\n")
                
                best_dynamic = max(self.results, key=lambda x: x['total_pnl'])
                best_native = max(self.native_trail_only_results, key=lambda x: x['total_pnl'])
                
                f.write(f"Best Dynamic Exit:       ${best_dynamic['total_pnl']:.2f} P&L | "
                       f"{best_dynamic['win_rate']:.1f}% WR | "
                       f"PF {best_dynamic['profit_factor']:.2f}\n")
                f.write(f"Best Native-Only:        ${best_native['total_pnl']:.2f} P&L | "
                       f"{best_native['win_rate']:.1f}% WR | "
                       f"PF {best_native['profit_factor']:.2f}\n\n")
                
                winner = "Dynamic exits" if best_dynamic['total_pnl'] > best_native['total_pnl'] else "Native-trail-only"
                f.write(f"WINNER: {winner} performed better\n")
        
        logging.info(f"Saved summary report: {summary_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Consolidated parameter optimizer for day trading"
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick mode (~12-16 tests)'
    )
    parser.add_argument(
        '--signals', 
        type=str,
        default='backtester/signals_to_test.txt',
        help='Path to signals file'
    )
    
    args = parser.parse_args()
    
    optimizer = ParameterOptimizer(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    await optimizer.run_optimization()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nOptimization interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
