#!/usr/bin/env python3
"""
parameter_optimizer_unified.py - UNIFIED SUPER OPTIMIZER
Automatically detects 0DTE vs regular signals and applies appropriate parameter grids
Handles mixed signal files intelligently
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
    Handles both 0DTE and regular signals in the same file
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/unified_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("🚀 UNIFIED ParameterOptimizer initialized")
        logging.info(f"📊 Signals file: {self.signals_file}")
        logging.info(f"⚡ Quick mode: {quick_mode}")
        logging.info(f"📂 Output dir: {self.output_dir}")
        
        # Load parameter grids
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
        self.signal_breakdown = {'0dte': [], 'regular': []}
    
    def get_0dte_quick_grid(self):
        """Quick test grid for 0DTE - tight parameters for 15-60 min holds"""
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
    
    def get_0dte_full_grid(self):
        """Full test grid for 0DTE - comprehensive testing"""
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
    
    def get_regular_quick_grid(self):
        """Quick test grid for regular options - wider parameters"""
        return {
            'breakeven_trigger_percent': [7, 10, 12],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [8, 10],
            'atr_period': [14],
            'atr_multiplier': [1.5],
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
        """Full test grid for regular options - comprehensive testing"""
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
        """
        Classify signal as 0DTE or regular based on expiry date
        
        Args:
            signal: Parsed signal dict with 'expiry' in YYYYMMDD format
        
        Returns:
            '0dte' or 'regular'
        """
        try:
            # Parse signal timestamp and expiry
            signal_time = datetime.strptime(signal['timestamp'], '%Y-%m-%d %H:%M:%S')
            expiry_date = datetime.strptime(signal['expiry'], '%Y%m%d')
            
            # Calculate days to expiry
            days_diff = (expiry_date.date() - signal_time.date()).days
            
            # 0DTE = same day expiry
            if days_diff == 0:
                return '0dte'
            else:
                return 'regular'
                
        except Exception as e:
            logging.warning(f"Could not classify signal: {e}. Defaulting to 'regular'")
            return 'regular'
    
    def load_and_classify_signals(self) -> Dict[str, List]:
        """
        Load signals from file and classify them as 0DTE or regular
        
        Returns:
            Dict with '0dte' and 'regular' keys containing lists of signals
        """
        from services.signal_parser import SignalParser
        
        config = Config()
        parser = SignalParser(config)
        
        # Create default profile for parsing
        default_profile = config.profiles[0] if config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True,
            'buzzwords_buy': [],
            'buzzwords_sell': [],
            'channel_id': 'optimizer'
        }
        
        classified = {'0dte': [], 'regular': []}
        
        if not self.signals_file.exists():
            logging.error(f"Signals file not found: {self.signals_file}")
            return classified
        
        with open(self.signals_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('Trader:') or line.startswith('Format:'):
                    continue
                
                # Parse signal
                if '|' in line:
                    parts = line.split('|')
                    timestamp_str = parts[0].strip()
                    channel = parts[1].strip()
                    signal_text = parts[2].strip()
                else:
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    channel = 'test_server'
                    signal_text = line
                
                # ✅ FIXED: Pass profile object, not channel string
                parsed = parser.parse_signal(signal_text, default_profile)
                
                if parsed:
                    parsed['timestamp'] = timestamp_str
                    
                    # Classify and add to appropriate list
                    signal_type = self.classify_signal(parsed)
                    classified[signal_type].append(parsed)
        
        logging.info(f"📊 Signal Classification:")
        logging.info(f"   0DTE signals: {len(classified['0dte'])}")
        logging.info(f"   Regular signals: {len(classified['regular'])}")
        
        return classified
    
    async def run_optimization(self):
        """Main optimization loop - handles mixed signals intelligently"""
        logging.info("=" * 80)
        logging.info("UNIFIED PARAMETER OPTIMIZATION")
        logging.info("=" * 80)
        
        # Load and classify signals
        classified_signals = self.load_and_classify_signals()
        
        total_signals = len(classified_signals['0dte']) + len(classified_signals['regular'])
        if total_signals == 0:
            logging.error("No valid signals found!")
            return
        
        # Generate parameter combinations for each type
        test_queue = []
        test_num = 1
        
        # Add 0DTE tests if we have 0DTE signals
        if classified_signals['0dte']:
            grid_0dte = self.param_grids['0dte']
            keys = sorted(grid_0dte.keys())
            combinations_0dte = [dict(zip(keys, v)) for v in product(*[grid_0dte[k] for k in keys])]
            
            logging.info(f"📋 0DTE Testing: {len(combinations_0dte)} parameter combinations")
            
            for params in combinations_0dte:
                test_queue.append({
                    'test_num': test_num,
                    'params': params,
                    'signal_type': '0dte'
                })
                test_num += 1
        
        # Add regular tests if we have regular signals
        if classified_signals['regular']:
            grid_regular = self.param_grids['regular']
            keys = sorted(grid_regular.keys())
            combinations_regular = [dict(zip(keys, v)) for v in product(*[grid_regular[k] for k in keys])]
            
            logging.info(f"📋 Regular Testing: {len(combinations_regular)} parameter combinations")
            
            for params in combinations_regular:
                test_queue.append({
                    'test_num': test_num,
                    'params': params,
                    'signal_type': 'regular'
                })
                test_num += 1
        
        total_tests = len(test_queue)
        logging.info(f"🎯 Total tests to run: {total_tests}")
        logging.info("=" * 80)
        
        # Run all tests
        for test in test_queue:
            result = await self.run_single_test(
                test['test_num'],
                test['params'],
                test['signal_type'],
                total_tests
            )
            if result:
                self.results.append(result)
        
        # Generate reports
        self.generate_reports()
        
        logging.info("=" * 80)
        logging.info("✅ OPTIMIZATION COMPLETE")
        logging.info("=" * 80)
        logging.info(f"📂 Results saved in: {self.output_dir}")
        
        if self.results:
            # Show best result for each type
            results_0dte = [r for r in self.results if r['signal_type'] == '0dte']
            results_regular = [r for r in self.results if r['signal_type'] == 'regular']
            
            if results_0dte:
                best_0dte = max(results_0dte, key=lambda x: x['total_pnl'])
                logging.info(f"\n🏆 Best 0DTE Configuration:")
                logging.info(f"   P&L: ${best_0dte['total_pnl']:,.2f}")
                logging.info(f"   Win Rate: {best_0dte['win_rate']:.1f}%")
                logging.info(f"   Profit Factor: {best_0dte['profit_factor']:.2f}")
            
            if results_regular:
                best_regular = max(results_regular, key=lambda x: x['total_pnl'])
                logging.info(f"\n🏆 Best Regular Configuration:")
                logging.info(f"   P&L: ${best_regular['total_pnl']:,.2f}")
                logging.info(f"   Win Rate: {best_regular['win_rate']:.1f}%")
                logging.info(f"   Profit Factor: {best_regular['profit_factor']:.2f}")
    
    async def run_single_test(self, test_num: int, params: Dict, signal_type: str, total_tests: int) -> Dict:
        """Run a single backtest with specific parameters"""
        test_name = f"test_{test_num:04d}_{signal_type}"
        
        logging.info(f"\n[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Type: {signal_type.upper()} | "
                    f"Breakeven: {params['breakeven_trigger_percent']}% | "
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
                # Add parameter details and signal type to results
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
        """Generate comprehensive summary reports"""
        if not self.results:
            logging.warning("No results to report!")
            return
        
        # Save detailed results CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "unified_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"📊 Saved detailed results: {csv_path}")
        
        # Generate text summary
        summary_path = self.output_dir / "unified_optimization_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("UNIFIED PARAMETER OPTIMIZATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall stats
            f.write(f"Total tests run: {len(self.results)}\n")
            f.write(f"Quick mode: {self.quick_mode}\n\n")
            
            # Separate results by type
            results_0dte = [r for r in self.results if r['signal_type'] == '0dte']
            results_regular = [r for r in self.results if r['signal_type'] == 'regular']
            
            # 0DTE Results
            if results_0dte:
                f.write("=" * 80 + "\n")
                f.write("0DTE SIGNALS RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                best_0dte = max(results_0dte, key=lambda x: x['total_pnl'])
                
                f.write(f"Tests run: {len(results_0dte)}\n")
                f.write(f"Profitable configs: {sum(1 for r in results_0dte if r['total_pnl'] > 0)}\n\n")
                
                f.write("🏆 BEST 0DTE CONFIGURATION:\n")
                f.write(f"  P&L: ${best_0dte['total_pnl']:,.2f}\n")
                f.write(f"  Win Rate: {best_0dte['win_rate']:.1f}%\n")
                f.write(f"  Profit Factor: {best_0dte['profit_factor']:.2f}\n")
                f.write(f"  Avg Minutes Held: {best_0dte.get('avg_minutes_held', 0):.0f}\n\n")
                
                f.write("  Parameters:\n")
                f.write(f"    Breakeven Trigger: {best_0dte['breakeven_trigger_percent']}%\n")
                f.write(f"    Trail Method: {best_0dte['trail_method']}\n")
                f.write(f"    Pullback %: {best_0dte['pullback_percent']}%\n")
                f.write(f"    Native Trail: {best_0dte['native_trail_percent']}%\n\n")
            
            # Regular Results
            if results_regular:
                f.write("=" * 80 + "\n")
                f.write("REGULAR SIGNALS RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                best_regular = max(results_regular, key=lambda x: x['total_pnl'])
                
                f.write(f"Tests run: {len(results_regular)}\n")
                f.write(f"Profitable configs: {sum(1 for r in results_regular if r['total_pnl'] > 0)}\n\n")
                
                f.write("🏆 BEST REGULAR CONFIGURATION:\n")
                f.write(f"  P&L: ${best_regular['total_pnl']:,.2f}\n")
                f.write(f"  Win Rate: {best_regular['win_rate']:.1f}%\n")
                f.write(f"  Profit Factor: {best_regular['profit_factor']:.2f}\n")
                f.write(f"  Avg Minutes Held: {best_regular.get('avg_minutes_held', 0):.0f}\n\n")
                
                f.write("  Parameters:\n")
                f.write(f"    Breakeven Trigger: {best_regular['breakeven_trigger_percent']}%\n")
                f.write(f"    Trail Method: {best_regular['trail_method']}\n")
                f.write(f"    Pullback %: {best_regular['pullback_percent']}%\n")
                f.write(f"    Native Trail: {best_regular['native_trail_percent']}%\n")
                f.write(f"    PSAR Enabled: {best_regular['psar_enabled']}\n")
                f.write(f"    RSI Hook Enabled: {best_regular['rsi_hook_enabled']}\n\n")
        
        logging.info(f"📋 Saved summary report: {summary_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified parameter optimization for both 0DTE and regular signals"
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
        help='Path to custom parameter grid JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ParameterOptimizerUnified(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    # Load custom params if provided
    if args.params:
        with open(args.params, 'r') as f:
            custom_grids = json.load(f)
        
        if '0dte_params' in custom_grids:
            optimizer.param_grids['0dte'] = custom_grids['0dte_params']
            logging.info(f"Loaded custom 0DTE grid from {args.params}")
        
        if 'regular_params' in custom_grids:
            optimizer.param_grids['regular'] = custom_grids['regular_params']
            logging.info(f"Loaded custom regular grid from {args.params}")
    
    # Run optimization
    await optimizer.run_optimization()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)
