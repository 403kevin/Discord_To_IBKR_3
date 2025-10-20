#!/usr/bin/env python3
"""
parameter_optimizer_unified.py - SUPER OPTIMIZER
Intelligently tests BOTH 0DTE and regular option parameters in one run
Detects signal type and applies appropriate parameter ranges
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


class UnifiedParameterOptimizer:
    """
    SUPER OPTIMIZER - Tests both 0DTE and regular option parameter ranges
    Automatically detects signal type and applies appropriate parameters
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("🚀 UNIFIED SUPER OPTIMIZER initialized")
        logging.info(f"📊 Signals file: {self.signals_file}")
        logging.info(f"⚡ Quick mode: {quick_mode}")
        logging.info(f"📂 Output dir: {self.output_dir}")
        
        # Load and categorize signals
        self.signals_0dte = []
        self.signals_regular = []
        self._categorize_signals()
        
        # Define parameter grids for each type
        self.param_grid_0dte = self.get_0dte_grid(quick_mode)
        self.param_grid_regular = self.get_regular_grid(quick_mode)
        
        self.results = []
    
    def _categorize_signals(self):
        """Separate signals into 0DTE and regular based on ticker"""
        logging.info("\n" + "="*80)
        logging.info("📋 CATEGORIZING SIGNALS BY TYPE")
        logging.info("="*80)
        
        if not self.signals_file.exists():
            logging.error(f"❌ Signals file not found: {self.signals_file}")
            return
        
        with open(self.signals_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse signal to get ticker
                parts = line.split('|')
                if len(parts) >= 3:
                    signal_text = parts[2].strip()
                else:
                    signal_text = line
                
                # Check if it's SPX/SPY (0DTE indicators)
                ticker_upper = signal_text.split()[0].upper()
                
                if ticker_upper in ['SPX', 'SPY', 'SPXW']:
                    self.signals_0dte.append(line)
                else:
                    self.signals_regular.append(line)
        
        logging.info(f"✅ Found {len(self.signals_0dte)} 0DTE signals (SPX/SPY)")
        logging.info(f"✅ Found {len(self.signals_regular)} regular option signals")
        logging.info("="*80)
    
def get_0dte_grid(self, quick_mode):
        """Parameter grid for 0DTE options (ultra-short holds)"""
        if quick_mode:
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
        else:
            # FOCUSED FULL GRID for 0DTE (50 combinations vs 100)
            return {
                'breakeven_trigger_percent': [5, 7, 10, 15],
                'trail_method': ['pullback_percent'],
                'pullback_percent': [10, 15, 20],
                'atr_period': [14],
                'atr_multiplier': [1.5],
                'native_trail_percent': [20, 25, 30],
                'psar_enabled': [False],
                'psar_start': [0.02],
                'psar_increment': [0.02],
                'psar_max': [0.2],
                'rsi_hook_enabled': [False],
                'rsi_period': [14],
                'rsi_overbought': [70],
                'rsi_oversold': [30]
            }
            # 4 (breakeven) × 3 (pullback) × 3 (native) = 36 tests
    
    def get_regular_grid(self, quick_mode):
        """Parameter grid for regular options (multi-hour holds)"""
        if quick_mode:
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
        else:
            # FOCUSED FULL GRID for regular (180 combinations vs 2,880)
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
            # 5 (breakeven) × 2 (trail_method) × 3 (pullback) × 2 (atr_multi) × 3 (native) = 180 tests
    
    def load_custom_params(self, params_file):
        """Load custom parameter grids from JSON file"""
        logging.info(f"📋 Loading custom parameters from {params_file}")
        
        with open(params_file, 'r') as f:
            custom_grids = json.load(f)
        
        if '0dte_params' in custom_grids:
            self.param_grid_0dte = custom_grids['0dte_params']
            logging.info("✅ Loaded custom 0DTE parameters")
        
        if 'regular_params' in custom_grids:
            self.param_grid_regular = custom_grids['regular_params']
            logging.info("✅ Loaded custom regular parameters")
    
    async def run_optimization(self):
        """Run optimization for both signal types"""
        logging.info("\n" + "="*100)
        logging.info("🚀 STARTING UNIFIED OPTIMIZATION")
        logging.info("="*100)
        
        all_results = []
        
        # Test 0DTE signals if we have any
        if self.signals_0dte:
            logging.info("\n" + "🎯"*40)
            logging.info("TESTING 0DTE SIGNALS (SPX/SPY)")
            logging.info("🎯"*40)
            
            results_0dte = await self._optimize_signal_set(
                self.signals_0dte,
                self.param_grid_0dte,
                signal_type="0DTE"
            )
            all_results.extend(results_0dte)
        
        # Test regular signals if we have any
        if self.signals_regular:
            logging.info("\n" + "📈"*40)
            logging.info("TESTING REGULAR OPTION SIGNALS")
            logging.info("📈"*40)
            
            results_regular = await self._optimize_signal_set(
                self.signals_regular,
                self.param_grid_regular,
                signal_type="REGULAR"
            )
            all_results.extend(results_regular)
        
        # Save combined results
        self._save_unified_results(all_results)
        
        logging.info("\n" + "="*100)
        logging.info("✅ UNIFIED OPTIMIZATION COMPLETE")
        logging.info("="*100)
    
    async def _optimize_signal_set(self, signals: List[str], param_grid: Dict, signal_type: str):
        """Optimize a specific set of signals with their appropriate parameter grid"""
        
        # Create temporary signals file
        temp_signals_file = self.output_dir / f"temp_signals_{signal_type.lower()}.txt"
        with open(temp_signals_file, 'w') as f:
            for signal in signals:
                f.write(signal + '\n')
        
        # Generate parameter combinations
        param_keys = list(param_grid.keys())
        param_values = [param_grid[key] for key in param_keys]
        combinations = list(product(*param_values))
        
        total_tests = len(combinations)
        logging.info(f"\n📊 Testing {total_tests} parameter combinations for {signal_type}")
        logging.info(f"⏱️  Estimated time: {total_tests * 5 / 60:.1f} minutes")
        
        results = []
        
        for test_num, combo in enumerate(combinations, 1):
            params = dict(zip(param_keys, combo))
            test_name = f"{signal_type}_test_{test_num:04d}"
            
            logging.info(f"\n{'='*100}")
            logging.info(f"[{test_num}/{total_tests}] Running: {test_name}")
            logging.info(f"{'='*100}")
            
            # Log parameters
            logging.info(f"Breakeven: {params['breakeven_trigger_percent']}%")
            logging.info(f"Trail: {params['trail_method']}")
            if params['trail_method'] == 'pullback_percent':
                logging.info(f"Pullback: {params['pullback_percent']}%")
            else:
                logging.info(f"ATR: {params['atr_period']}p × {params['atr_multiplier']}")
            logging.info(f"Native: {params['native_trail_percent']}%")
            logging.info(f"PSAR: {params['psar_enabled']} | RSI: {params['rsi_hook_enabled']}")
            
            # Run backtest
            engine = BacktestEngine(
                str(temp_signals_file),
                "backtester/historical_data"
            )
            
            backtest_results = engine.run_simulation(params)
            
            # Store results
            result = {
                'test_name': test_name,
                'signal_type': signal_type,
                **backtest_results,
                **params
            }
            results.append(result)
            
            # Log quick summary
            logging.info(f"✅ P&L: ${backtest_results['total_pnl']:.2f} | "
                        f"Win Rate: {backtest_results['win_rate']:.1f}% | "
                        f"PF: {backtest_results['profit_factor']:.2f}")
        
        # Clean up temp file
        temp_signals_file.unlink()
        
        return results
    
    def _save_unified_results(self, all_results: List[Dict]):
        """Save combined results for both signal types"""
        
        if not all_results:
            logging.warning("⚠️  No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Save detailed CSV
        csv_path = self.output_dir / "unified_results.csv"
        df.to_csv(csv_path, index=False)
        logging.info(f"\n📊 Saved detailed results to {csv_path}")
        
        # Create summary report
        self._create_unified_summary(df)
    
    def _create_unified_summary(self, df: pd.DataFrame):
        """Create comprehensive summary comparing both signal types"""
        
        summary_path = self.output_dir / "unified_optimization_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("="*100 + "\n")
            f.write("UNIFIED PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Tests both 0DTE and Regular option parameters\n")
            f.write("="*100 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-"*100 + "\n")
            f.write(f"Total Tests: {len(df)}\n")
            f.write(f"Profitable Configs: {len(df[df['total_pnl'] > 0])} ({len(df[df['total_pnl'] > 0])/len(df)*100:.1f}%)\n")
            f.write(f"Average P&L: ${df['total_pnl'].mean():.2f}\n")
            f.write(f"Best P&L: ${df['total_pnl'].max():.2f}\n")
            f.write(f"Worst P&L: ${df['total_pnl'].min():.2f}\n\n")
            
            # Breakdown by signal type
            for signal_type in df['signal_type'].unique():
                df_type = df[df['signal_type'] == signal_type]
                
                f.write(f"\n{signal_type} SIGNALS BREAKDOWN:\n")
                f.write("-"*100 + "\n")
                f.write(f"Tests Run: {len(df_type)}\n")
                f.write(f"Profitable: {len(df_type[df_type['total_pnl'] > 0])} ({len(df_type[df_type['total_pnl'] > 0])/len(df_type)*100:.1f}%)\n")
                f.write(f"Average P&L: ${df_type['total_pnl'].mean():.2f}\n")
                f.write(f"Average Win Rate: {df_type['win_rate'].mean():.1f}%\n")
                f.write(f"Average Profit Factor: {df_type['profit_factor'].mean():.2f}\n\n")
                
                # Top 5 for this type
                top_5 = df_type.nlargest(5, 'total_pnl')
                f.write(f"TOP 5 {signal_type} CONFIGURATIONS:\n")
                for i, (idx, row) in enumerate(top_5.iterrows(), 1):
                    f.write(f"\n#{i}. {row['test_name']}\n")
                    f.write(f"   Win Rate: {row['win_rate']:.1f}% | P&L: ${row['total_pnl']:.2f} | PF: {row['profit_factor']:.2f}\n")
                    f.write(f"   Breakeven: {row['breakeven_trigger_percent']}% | ")
                    if row['trail_method'] == 'pullback_percent':
                        f.write(f"Pullback: {row['pullback_percent']}% | ")
                    else:
                        f.write(f"ATR: {row['atr_period']}p×{row['atr_multiplier']} | ")
                    f.write(f"Native: {row['native_trail_percent']}%\n")
            
            # Overall top 10
            f.write("\n" + "="*100 + "\n")
            f.write("TOP 10 OVERALL CONFIGURATIONS (All Signal Types):\n")
            f.write("="*100 + "\n\n")
            
            top_10 = df.nlargest(10, 'total_pnl')
            for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                f.write(f"#{i}. {row['test_name']} [{row['signal_type']}]\n")
                f.write(f"   Win Rate: {row['win_rate']:.1f}% | P&L: ${row['total_pnl']:.2f} | PF: {row['profit_factor']:.2f}\n")
                f.write(f"   Breakeven: {row['breakeven_trigger_percent']}% | ")
                if row['trail_method'] == 'pullback_percent':
                    f.write(f"Pullback: {row['pullback_percent']}% | ")
                else:
                    f.write(f"ATR: {row['atr_period']}p×{row['atr_multiplier']} | ")
                f.write(f"Native: {row['native_trail_percent']}%\n\n")
        
        logging.info(f"📋 Saved unified summary to {summary_path}")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified parameter optimization for both 0DTE and regular options"
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
        help='Path to signals file (will auto-categorize)'
    )
    parser.add_argument(
        '--params',
        type=str,
        help='Path to custom unified parameter grid JSON file'
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = UnifiedParameterOptimizer(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    # Load custom params if provided
    if args.params:
        optimizer.load_custom_params(args.params)
    
    # Run optimization
    await optimizer.run_optimization()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)
