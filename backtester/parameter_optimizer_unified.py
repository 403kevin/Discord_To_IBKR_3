#!/usr/bin/env python3
"""
parameter_optimizer_unified.py - COMPLETE FIXED VERSION
Auto-detects 0DTE vs Regular signals and applies appropriate parameters
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
        
        self.results = []
        
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
    
    def get_0dte_quick_grid(self) -> Dict:
        """Quick test parameters for 0DTE signals"""
        return {
            'breakeven_trigger_percent': [7, 10],
            'trail_method': ['pullback_percent'],
            'pullback_percent': [10, 15],
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'native_trail_percent': [20, 25],
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def get_0dte_full_grid(self) -> Dict:
        """Full test parameters for 0DTE signals"""
        return {
            'breakeven_trigger_percent': [5, 7, 10, 12, 15],
            'trail_method': ['pullback_percent'],
            'pullback_percent': [8, 10, 12, 15, 20],
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'native_trail_percent': [15, 20, 25, 30],
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30]
        }
    
    def get_regular_quick_grid(self) -> Dict:
        """Quick test parameters for regular signals"""
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
    
    def get_regular_full_grid(self) -> Dict:
        """Full test parameters for regular signals"""
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
    
    def classify_signal(self, parsed_signal: Dict) -> str:
        """Classify signal as 0dte or regular based on ticker and expiry"""
        ticker = parsed_signal.get('ticker', '').upper()
        
        # 0DTE tickers
        if ticker in ['SPX', 'SPY', 'SPXW']:
            # Check if expiry is today (0DTE)
            try:
                from datetime import datetime
                expiry_date = parsed_signal.get('expiry_date')
                if isinstance(expiry_date, str):
                    expiry = datetime.strptime(expiry_date, '%Y%m%d').date()
                else:
                    expiry = expiry_date.date() if hasattr(expiry_date, 'date') else expiry_date
                
                timestamp = parsed_signal.get('timestamp')
                if isinstance(timestamp, str):
                    signal_date = datetime.strptime(timestamp.split()[0], '%Y-%m-%d').date()
                else:
                    signal_date = timestamp.date()
                
                # If expiry is same day as signal, it's 0DTE
                if expiry == signal_date:
                    logging.debug(f"Classified {ticker} as 0DTE (same-day expiry)")
                    return '0dte'
                else:
                    logging.debug(f"Classified {ticker} as regular (multi-day expiry)")
                    return 'regular'
            except Exception as e:
                logging.warning(f"Error classifying {ticker}: {e}. Defaulting to regular.")
                return 'regular'
        
        # All other tickers are regular
        logging.debug(f"Classified {ticker} as regular (non-index)")
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
        """Generate DETAILED summary reports matching QRG standards"""
        if not self.results:
            logging.warning("No results to report!")
            return
        
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "unified_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved detailed results: {csv_path}")
        
        summary_path = self.output_dir / "unified_optimization_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("UNIFIED PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Tests both 0DTE and Regular option parameters\n")
            f.write("=" * 100 + "\n\n")
            
            # OVERALL PERFORMANCE
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 100 + "\n")
            total_tests = len(self.results)
            profitable = sum(1 for r in self.results if r['total_pnl'] > 0)
            avg_pnl = np.mean([r['total_pnl'] for r in self.results])
            best_pnl = max([r['total_pnl'] for r in self.results])
            worst_pnl = min([r['total_pnl'] for r in self.results])
            
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Profitable Configs: {profitable} ({profitable/total_tests*100:.1f}%)\n")
            f.write(f"Average P&L: ${avg_pnl:.2f}\n")
            f.write(f"Best P&L: ${best_pnl:.2f}\n")
            f.write(f"Worst P&L: ${worst_pnl:.2f}\n")
            f.write(f"Quick Mode: {self.quick_mode}\n\n")
            
            # SEPARATE BY SIGNAL TYPE
            results_0dte = [r for r in self.results if r['signal_type'] == '0dte']
            results_regular = [r for r in self.results if r['signal_type'] == 'regular']
            
            # 0DTE SECTION
            if results_0dte:
                f.write("=" * 100 + "\n")
                f.write("0DTE SIGNALS BREAKDOWN:\n")
                f.write("-" * 100 + "\n")
                
                profitable_0dte = sum(1 for r in results_0dte if r['total_pnl'] > 0)
                avg_pnl_0dte = np.mean([r['total_pnl'] for r in results_0dte])
                avg_wr_0dte = np.mean([r['win_rate'] for r in results_0dte])
                avg_pf_0dte = np.mean([r['profit_factor'] for r in results_0dte])
                
                f.write(f"Tests Run: {len(results_0dte)}\n")
                f.write(f"Profitable: {profitable_0dte} ({profitable_0dte/len(results_0dte)*100:.1f}%)\n")
                f.write(f"Average P&L: ${avg_pnl_0dte:.2f}\n")
                f.write(f"Average Win Rate: {avg_wr_0dte:.1f}%\n")
                f.write(f"Average Profit Factor: {avg_pf_0dte:.2f}\n\n")
                
                # TOP 5 0DTE CONFIGS
                f.write("TOP 5 0DTE CONFIGURATIONS:\n\n")
                sorted_0dte = sorted(results_0dte, key=lambda x: x['total_pnl'], reverse=True)[:5]
                for idx, config in enumerate(sorted_0dte, 1):
                    f.write(f"#{idx}. {config['test_name']}\n")
                    f.write(f"   Win Rate: {config['win_rate']:.1f}% | ")
                    f.write(f"P&L: ${config['total_pnl']:.2f} | ")
                    f.write(f"PF: {config['profit_factor']:.2f}\n")
                    f.write(f"   Breakeven: {config['breakeven_trigger_percent']}% | ")
                    f.write(f"Pullback: {config['pullback_percent']}% | ")
                    f.write(f"Native: {config['native_trail_percent']}%\n")
                    f.write(f"   Avg Hold: {config.get('avg_minutes_held', 0):.0f} min | ")
                    f.write(f"Return: {config['return_pct']:.2f}%\n")
                    if 'exit_reasons' in config:
                        f.write(f"   Exits: {config['exit_reasons']}\n")
                    f.write("\n")
            
            # REGULAR SECTION
            if results_regular:
                f.write("=" * 100 + "\n")
                f.write("REGULAR SIGNALS BREAKDOWN:\n")
                f.write("-" * 100 + "\n")
                
                profitable_reg = sum(1 for r in results_regular if r['total_pnl'] > 0)
                avg_pnl_reg = np.mean([r['total_pnl'] for r in results_regular])
                avg_wr_reg = np.mean([r['win_rate'] for r in results_regular])
                avg_pf_reg = np.mean([r['profit_factor'] for r in results_regular])
                
                f.write(f"Tests Run: {len(results_regular)}\n")
                f.write(f"Profitable: {profitable_reg} ({profitable_reg/len(results_regular)*100:.1f}%)\n")
                f.write(f"Average P&L: ${avg_pnl_reg:.2f}\n")
                f.write(f"Average Win Rate: {avg_wr_reg:.1f}%\n")
                f.write(f"Average Profit Factor: {avg_pf_reg:.2f}\n\n")
                
                # TOP 5 REGULAR CONFIGS
                f.write("TOP 5 REGULAR CONFIGURATIONS:\n\n")
                sorted_reg = sorted(results_regular, key=lambda x: x['total_pnl'], reverse=True)[:5]
                for idx, config in enumerate(sorted_reg, 1):
                    f.write(f"#{idx}. {config['test_name']}\n")
                    f.write(f"   Win Rate: {config['win_rate']:.1f}% | ")
                    f.write(f"P&L: ${config['total_pnl']:.2f} | ")
                    f.write(f"PF: {config['profit_factor']:.2f}\n")
                    f.write(f"   Breakeven: {config['breakeven_trigger_percent']}% | ")
                    f.write(f"Trail: {config['trail_method']} | ")
                    if config['trail_method'] == 'pullback_percent':
                        f.write(f"Pullback: {config['pullback_percent']}% | ")
                    else:
                        f.write(f"ATR: {config['atr_period']}p x {config['atr_multiplier']} | ")
                    f.write(f"Native: {config['native_trail_percent']}%\n")
                    f.write(f"   PSAR: {config['psar_enabled']} | RSI: {config['rsi_hook_enabled']} | ")
                    f.write(f"Avg Hold: {config.get('avg_minutes_held', 0):.0f} min\n")
                    f.write(f"   Return: {config['return_pct']:.2f}%\n")
                    if 'exit_reasons' in config:
                        f.write(f"   Exits: {config['exit_reasons']}\n")
                    f.write("\n")
            
            # COMPARISON
            if results_0dte and results_regular:
                f.write("=" * 100 + "\n")
                f.write("SIGNAL TYPE COMPARISON:\n")
                f.write("-" * 100 + "\n")
                f.write(f"0DTE:    {len(results_0dte)} tests | ")
                f.write(f"Avg P&L ${avg_pnl_0dte:.2f} | ")
                f.write(f"{avg_wr_0dte:.1f}% WR | ")
                f.write(f"{avg_pf_0dte:.2f} PF\n")
                f.write(f"Regular: {len(results_regular)} tests | ")
                f.write(f"Avg P&L ${avg_pnl_reg:.2f} | ")
                f.write(f"{avg_wr_reg:.1f}% WR | ")
                f.write(f"{avg_pf_reg:.2f} PF\n\n")
                
                winner = "0DTE" if avg_pnl_0dte > avg_pnl_reg else "Regular"
                f.write(f"WINNER: {winner} signals performed better on average\n")
        
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
