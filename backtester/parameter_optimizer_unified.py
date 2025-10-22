#!/usr/bin/env python3
"""
parameter_optimizer_unified.py - FIXED FOR WINDOWS
Removed emoji characters that cause encoding errors on Windows
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
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


class ParameterOptimizerUnified:
    """
    UNIFIED OPTIMIZER - Auto-detects signal types and applies correct parameters
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/unified_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("UNIFIED ParameterOptimizer initialized")
        logging.info(f"Signals file: {self.signals_file}")
        logging.info(f"Quick mode: {quick_mode}")
        logging.info(f"Output dir: {self.output_dir}")
        
        if quick_mode:
            self.param_grids = {
                '0dte': self.get_0dte_quick_grid(),
                'regular': self.get_regular_quick_grid()
            }
        else:
            self.param_grids = {
                '0dte': self.get_0dte_full_grid(),
                'regular': self.get_regular_full_grid()
            }
        
        self.results = []
    
    def get_0dte_quick_grid(self):
        """Quick test grid for 0DTE"""
        return {
            'breakeven_trigger_percent': [7, 10],
            'trail_method': ['pullback_percent'],
            'pullback_percent': [10, 15],
            'native_trail_percent': [20, 25],
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def get_0dte_full_grid(self):
        """Full test grid for 0DTE"""
        return {
            'breakeven_trigger_percent': [5, 7, 10, 12],
            'trail_method': ['pullback_percent'],
            'pullback_percent': [8, 10, 12, 15],
            'native_trail_percent': [20, 25, 30],
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'psar_enabled': [True, False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def get_regular_quick_grid(self):
        """Quick test grid for regular options"""
        return {
            'breakeven_trigger_percent': [7, 10, 12],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [10, 12],
            'atr_period': [14],
            'atr_multiplier': [1.5, 2.0],
            'native_trail_percent': [25, 30],
            'psar_enabled': [True, False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def get_regular_full_grid(self):
        """Full test grid for regular options"""
        return {
            'breakeven_trigger_percent': [5, 7, 10, 12, 15],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [7, 10, 12, 15],
            'atr_period': [10, 14, 20],
            'atr_multiplier': [1.0, 1.5, 2.0],
            'native_trail_percent': [20, 25, 30, 35],
            'psar_enabled': [True, False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [True, False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def classify_signal(self, signal: Dict) -> str:
        """Classify signal as 0DTE or regular"""
        try:
            signal_time = datetime.strptime(signal['timestamp'], '%Y-%m-%d %H:%M:%S')
            expiry_date = datetime.strptime(signal['expiry'], '%Y%m%d')
            days_diff = (expiry_date.date() - signal_time.date()).days
            return '0dte' if days_diff == 0 else 'regular'
        except Exception as e:
            logging.warning(f"Could not classify signal: {e}. Defaulting to regular.")
            return 'regular'
    
    def classify_signals_from_file(self) -> Dict[str, List[Dict]]:
        """Load and classify all signals"""
        from services.signal_parser import SignalParser
        
        config = Config()
        parser = SignalParser(config)
        
        default_profile = config.profiles[0] if config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True,
            'buzzwords_buy': [],
            'buzzwords_sell': [],
            'channel_id': 'backtest'
        }
        
        classified = {'0dte': [], 'regular': []}
        
        with open(self.signals_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Trader:') or line.startswith('Format:'):
                    continue
                
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    channel = parts[1].strip()
                    signal_text = parts[2].strip()
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    channel = 'test_server'
                    signal_text = line
                
                parsed = parser.parse_signal(signal_text, default_profile)
                
                if parsed:
                    parsed['timestamp'] = timestamp_str
                    signal_type = self.classify_signal(parsed)
                    classified[signal_type].append(parsed)
        
        logging.info(f"Signal Classification:")
        logging.info(f"   0DTE signals: {len(classified['0dte'])}")
        logging.info(f"   Regular signals: {len(classified['regular'])}")
        
        return classified
    
    def generate_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations"""
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combinations = [dict(zip(keys, combo)) for combo in product(*values)]
        return combinations
    
    async def run_optimization(self):
        """Run unified optimization"""
        logging.info("\n" + "="*80)
        logging.info("UNIFIED PARAMETER OPTIMIZATION START")
        logging.info("="*80)
        
        classified_signals = self.classify_signals_from_file()
        
        test_num = 0
        total_tests = 0
        
        for signal_type in ['0dte', 'regular']:
            if classified_signals[signal_type]:
                param_combos = self.generate_combinations(self.param_grids[signal_type])
                total_tests += len(param_combos)
        
        for signal_type in ['0dte', 'regular']:
            if not classified_signals[signal_type]:
                logging.info(f"\nNo {signal_type} signals found - skipping")
                continue
            
            logging.info(f"\n{'='*80}")
            logging.info(f"TESTING {signal_type.upper()} SIGNALS")
            logging.info(f"{'='*80}")
            
            param_combinations = self.generate_combinations(self.param_grids[signal_type])
            logging.info(f"Testing {len(param_combinations)} parameter combinations")
            
            for params in param_combinations:
                test_num += 1
                result = await self.run_single_test(test_num, params, signal_type, total_tests)
                if result:
                    self.results.append(result)
        
        self.generate_reports()
        
        logging.info("=" * 80)
        logging.info("OPTIMIZATION COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Results saved in: {self.output_dir}")
        
        if self.results:
            results_0dte = [r for r in self.results if r['signal_type'] == '0dte']
            results_regular = [r for r in self.results if r['signal_type'] == 'regular']
            
            if results_0dte:
                best_0dte = max(results_0dte, key=lambda x: x['total_pnl'])
                logging.info(f"\nBest 0DTE Configuration:")
                logging.info(f"   P&L: ${best_0dte['total_pnl']:,.2f}")
                logging.info(f"   Win Rate: {best_0dte['win_rate']:.1f}%")
                logging.info(f"   Profit Factor: {best_0dte['profit_factor']:.2f}")
            
            if results_regular:
                best_regular = max(results_regular, key=lambda x: x['total_pnl'])
                logging.info(f"\nBest Regular Configuration:")
                logging.info(f"   P&L: ${best_regular['total_pnl']:,.2f}")
                logging.info(f"   Win Rate: {best_regular['win_rate']:.1f}%")
                logging.info(f"   Profit Factor: {best_regular['profit_factor']:.2f}")
    
    async def run_single_test(self, test_num: int, params: Dict, signal_type: str, total_tests: int) -> Dict:
        """Run single backtest"""
        test_name = f"test_{test_num:04d}_{signal_type}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Type: {signal_type.upper()} | "
                    f"Breakeven: {params['breakeven_trigger_percent']}% | "
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
                    'signal_type': signal_type,
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
                    **params
                }
                
                if 'exit_reasons' in results:
                    summary['exit_reasons'] = results['exit_reasons']
                
                logging.info(f"  Results: {results['total_trades']} trades | "
                           f"${results['total_pnl']:.0f} P&L | "
                           f"{results['win_rate']:.1f}% WR")
                
                return summary
            else:
                logging.warning(f"  No results returned")
                return None
                
        except Exception as e:
            logging.error(f"  Error in {test_name}: {str(e)}")
            return None
    
    def generate_reports(self):
        """Generate summary reports - FIXED FOR WINDOWS"""
        if not self.results:
            logging.warning("No results to report!")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "unified_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved detailed results: {csv_path}")
        
        summary_path = self.output_dir / "unified_optimization_summary.txt"
        
        # FIX: Use UTF-8 encoding for Windows
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED PARAMETER OPTIMIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total tests run: {len(self.results)}\n")
            f.write(f"Quick mode: {self.quick_mode}\n\n")
            
            results_0dte = [r for r in self.results if r['signal_type'] == '0dte']
            results_regular = [r for r in self.results if r['signal_type'] == 'regular']
            
            if results_0dte:
                f.write("=" * 80 + "\n")
                f.write("0DTE SIGNALS RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                best_0dte = max(results_0dte, key=lambda x: x['total_pnl'])
                
                f.write(f"Tests run: {len(results_0dte)}\n")
                f.write(f"Profitable configs: {sum(1 for r in results_0dte if r['total_pnl'] > 0)}\n\n")
                
                f.write("BEST 0DTE CONFIGURATION:\n")
                f.write(f"  P&L: ${best_0dte['total_pnl']:,.2f}\n")
                f.write(f"  Win Rate: {best_0dte['win_rate']:.1f}%\n")
                f.write(f"  Profit Factor: {best_0dte['profit_factor']:.2f}\n")
                f.write(f"  Avg Minutes Held: {best_0dte.get('avg_minutes_held', 0):.0f}\n")
                f.write(f"  Return: {best_0dte['return_pct']:.2f}%\n\n")
                
                f.write("PARAMETERS:\n")
                f.write(f"  Breakeven Trigger: {best_0dte['breakeven_trigger_percent']}%\n")
                f.write(f"  Trail Method: {best_0dte['trail_method']}\n")
                f.write(f"  Pullback: {best_0dte['pullback_percent']}%\n")
                f.write(f"  Native Trail: {best_0dte['native_trail_percent']}%\n")
                f.write(f"  PSAR: {best_0dte['psar_enabled']}\n")
                f.write(f"  RSI Hook: {best_0dte['rsi_hook_enabled']}\n\n")
            
            if results_regular:
                f.write("=" * 80 + "\n")
                f.write("REGULAR SIGNALS RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                best_regular = max(results_regular, key=lambda x: x['total_pnl'])
                
                f.write(f"Tests run: {len(results_regular)}\n")
                f.write(f"Profitable configs: {sum(1 for r in results_regular if r['total_pnl'] > 0)}\n\n")
                
                f.write("BEST REGULAR CONFIGURATION:\n")
                f.write(f"  P&L: ${best_regular['total_pnl']:,.2f}\n")
                f.write(f"  Win Rate: {best_regular['win_rate']:.1f}%\n")
                f.write(f"  Profit Factor: {best_regular['profit_factor']:.2f}\n")
                f.write(f"  Avg Minutes Held: {best_regular.get('avg_minutes_held', 0):.0f}\n")
                f.write(f"  Return: {best_regular['return_pct']:.2f}%\n\n")
                
                f.write("PARAMETERS:\n")
                f.write(f"  Breakeven Trigger: {best_regular['breakeven_trigger_percent']}%\n")
                f.write(f"  Trail Method: {best_regular['trail_method']}\n")
                f.write(f"  Pullback: {best_regular['pullback_percent']}%\n")
                f.write(f"  ATR Period: {best_regular['atr_period']}\n")
                f.write(f"  ATR Multiplier: {best_regular['atr_multiplier']}\n")
                f.write(f"  Native Trail: {best_regular['native_trail_percent']}%\n")
                f.write(f"  PSAR: {best_regular['psar_enabled']}\n")
                f.write(f"  RSI Hook: {best_regular['rsi_hook_enabled']}\n\n")
        
        logging.info(f"Saved summary report: {summary_path}")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick test mode')
    parser.add_argument('--signals', default='backtester/signals_to_test.txt', help='Path to signals file')
    parser.add_argument('--params', help='Path to custom parameter grid JSON')
    args = parser.parse_args()
    
    optimizer = ParameterOptimizerUnified(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    if args.params:
        with open(args.params, 'r') as f:
            custom_grids = json.load(f)
            if '0dte_params' in custom_grids:
                optimizer.param_grids['0dte'] = custom_grids['0dte_params']
            if 'regular_params' in custom_grids:
                optimizer.param_grids['regular'] = custom_grids['regular_params']
    
    await optimizer.run_optimization()


if __name__ == "__main__":
    asyncio.run(main())
