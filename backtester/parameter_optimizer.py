#!/usr/bin/env python3
"""
parameter_optimizer.py - FIXED VERSION WITH DEBUG LOGGING
This version includes fixes for the optimization loop issue and comprehensive debugging
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
    FIXED VERSION: Automated parameter optimization with debug logging
    """
    
    def __init__(self, signals_file="backtester/signals_to_test.txt", quick_mode=False):
        self.signals_file = Path(signals_file)
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info("🔍 DEBUG: ParameterOptimizer initialized")
        logging.info(f"🔍 DEBUG: Signals file: {self.signals_file}")
        logging.info(f"🔍 DEBUG: Quick mode: {quick_mode}")
        logging.info(f"🔍 DEBUG: Output dir: {self.output_dir}")
        
        # Define parameter grid
        if quick_mode:
            self.param_grid = self.get_quick_grid()
        else:
            self.param_grid = self.get_full_grid()
        
        self.results = []
        self.config = Config()
        
        logging.info(f"🔍 DEBUG: Parameter grid keys: {list(self.param_grid.keys())}")
    
    def get_quick_grid(self) -> Dict[str, List[Any]]:
        """Quick grid for testing (16 combinations)"""
        logging.info("🔍 DEBUG: Building QUICK parameter grid")
        
        grid = {
            'breakeven_trigger_percent': [10, 15],
            'trail_method': ['atr', 'pullback_percent'],
            'pullback_percent': [10],  # Fixed for quick mode
            'atr_period': [14],  # Fixed for quick mode
            'atr_multiplier': [1.5],  # Fixed for quick mode
            'psar_enabled': [True, False],
            'psar_start': [0.02],  # Fixed for quick mode
            'psar_increment': [0.02],  # Fixed for quick mode
            'psar_max': [0.2],  # Fixed for quick mode
            'rsi_hook_enabled': [True, False],
            'rsi_period': [14],  # Fixed for quick mode
            'rsi_overbought': [70],  # Fixed for quick mode
            'rsi_oversold': [30],  # Fixed for quick mode
        }
        
        # Calculate total combinations
        total = 1
        for key, values in grid.items():
            total *= len(values)
        
        logging.info(f"🔍 DEBUG: Quick grid will generate {total} combinations")
        return grid
    
    def get_full_grid(self) -> Dict[str, List[Any]]:
        """Full grid for comprehensive testing"""
        logging.info("🔍 DEBUG: Building FULL parameter grid")
        
        grid = {
            'breakeven_trigger_percent': [5, 10, 15, 20],
            'trail_method': ['atr', 'pullback_percent'],
            'pullback_percent': [8, 10, 12, 15],
            'atr_period': [10, 14, 20],
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
            'psar_enabled': [True, False],
            'psar_start': [0.01, 0.02, 0.03],
            'psar_increment': [0.01, 0.02, 0.03],
            'psar_max': [0.1, 0.2, 0.3],
            'rsi_hook_enabled': [True, False],
            'rsi_period': [10, 14, 20],
            'rsi_overbought': [65, 70, 75],
            'rsi_oversold': [25, 30, 35],
        }
        
        # Calculate total combinations
        total = 1
        for key, values in grid.items():
            total *= len(values)
        
        logging.info(f"🔍 DEBUG: Full grid will generate {total} combinations")
        return grid
    
    def generate_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from the grid"""
        logging.info("🔍 DEBUG: generate_combinations() called")
        
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        logging.info(f"🔍 DEBUG: Grid has {len(keys)} parameters")
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        logging.info(f"🔍 DEBUG: Generated {len(combinations)} parameter combinations")
        
        if combinations:
            logging.info(f"🔍 DEBUG: First combination: {combinations[0]}")
            logging.info(f"🔍 DEBUG: Last combination: {combinations[-1]}")
        else:
            logging.error("❌ No combinations generated! Check parameter grid!")
        
        return combinations
    
    async def run_single_test(self, test_num: int, params: Dict, total_tests: int) -> Dict:
        """Run a single backtest with specific parameters"""
        test_name = f"test_{test_num:04d}"
        
        logging.info(f"\n🔍 DEBUG: Starting test {test_num}/{total_tests}")
        logging.info(f"[{test_num}/{total_tests}] Running: {test_name}")
        logging.info(f"  Breakeven: {params['breakeven_trigger_percent']}% | Trail: {params['trail_method']}")
        logging.info(f"  Pullback: {params['pullback_percent']}% | PSAR: {params['psar_enabled']} | RSI: {params['rsi_hook_enabled']}")
        
        try:
            # Create backtest engine with custom config
            logging.debug(f"🔍 DEBUG: Creating BacktestEngine for {test_name}")
            engine = BacktestEngine(
                signal_file_path=str(self.signals_file),
                data_folder_path="backtester/historical_data"
            )
            
            # Apply parameters to config
            logging.debug(f"🔍 DEBUG: Applying parameters to config")
            profile = engine.config.profiles[0] if engine.config.profiles else {}
            
            # Exit strategy parameters
            if 'exit_strategy' not in profile:
                profile['exit_strategy'] = {}
            
            profile['exit_strategy']['breakeven_trigger_percent'] = params['breakeven_trigger_percent'] / 100
            profile['exit_strategy']['trail_method'] = params['trail_method']
            
            if 'trail_settings' not in profile['exit_strategy']:
                profile['exit_strategy']['trail_settings'] = {}
            
            profile['exit_strategy']['trail_settings']['pullback_percent'] = params['pullback_percent'] / 100
            profile['exit_strategy']['trail_settings']['atr_period'] = params['atr_period']
            profile['exit_strategy']['trail_settings']['atr_multiplier'] = params['atr_multiplier']
            
            # Momentum exits
            if 'momentum_exits' not in profile['exit_strategy']:
                profile['exit_strategy']['momentum_exits'] = {}
            
            profile['exit_strategy']['momentum_exits']['psar_enabled'] = params['psar_enabled']
            
            if 'psar_settings' not in profile['exit_strategy']['momentum_exits']:
                profile['exit_strategy']['momentum_exits']['psar_settings'] = {}
            
            profile['exit_strategy']['momentum_exits']['psar_settings']['start'] = params['psar_start']
            profile['exit_strategy']['momentum_exits']['psar_settings']['increment'] = params['psar_increment']
            profile['exit_strategy']['momentum_exits']['psar_settings']['max'] = params['psar_max']
            
            profile['exit_strategy']['momentum_exits']['rsi_hook_enabled'] = params['rsi_hook_enabled']
            
            if 'rsi_settings' not in profile['exit_strategy']['momentum_exits']:
                profile['exit_strategy']['momentum_exits']['rsi_settings'] = {}
            
            profile['exit_strategy']['momentum_exits']['rsi_settings']['period'] = params['rsi_period']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['overbought_level'] = params['rsi_overbought']
            profile['exit_strategy']['momentum_exits']['rsi_settings']['oversold_level'] = params['rsi_oversold']
            
            # Run simulation
            logging.debug(f"🔍 DEBUG: Running simulation for {test_name}")
            engine.run_simulation()
            
            # Analyze results
            results_file = Path("backtester/backtest_results.csv")
            
            if results_file.exists():
                logging.debug(f"🔍 DEBUG: Analyzing results for {test_name}")
                summary = self.analyze_results(results_file, test_name, params)
                self.results.append(summary)
                logging.info(f"✅ Test {test_num} complete: P&L=${summary['total_pnl']:.2f}, WR={summary['win_rate']:.1f}%")
                return summary
            else:
                logging.warning(f"⚠️ No results file generated for {test_name}")
                return None
                
        except Exception as e:
            logging.error(f"❌ Test {test_name} failed: {e}", exc_info=True)
            return None
    
    def analyze_results(self, results_file: Path, test_name: str, params: Dict) -> Dict:
        """Analyze backtest results and generate metrics"""
        logging.debug(f"🔍 DEBUG: Analyzing results from {results_file}")
        
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
        """Generate comprehensive optimization report"""
        logging.info("\n🔍 DEBUG: Generating optimization report")
        
        if not self.results:
            logging.warning("⚠️ No results to report!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save all results
        full_results_file = self.output_dir / "all_results.csv"
        df.to_csv(full_results_file, index=False)
        logging.info(f"📊 Full results saved to {full_results_file}")
        
        # Generate summary
        summary_file = self.output_dir / "optimization_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("PARAMETER OPTIMIZATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            # Top 10 by P&L
            f.write("TOP 10 PARAMETER COMBINATIONS (by Total P&L):\n")
            f.write("-"*100 + "\n")
            
            df_sorted = df.sort_values('total_pnl', ascending=False).head(10)
            for idx, row in df_sorted.iterrows():
                f.write(f"\n#{df_sorted.index.get_loc(idx)+1}. {row.test_name}\n")
                f.write(f"   Win Rate: {row.win_rate:.1f}% | P&L: ${row.total_pnl:,.2f} | PF: {row.profit_factor:.2f}\n")
                f.write(f"   Breakeven: {row.breakeven_trigger_percent}% | Trail: {row.trail_method}\n")
                f.write(f"   Pullback: {row.pullback_percent}% | PSAR: {row.psar_enabled} | RSI: {row.rsi_hook_enabled}\n")
            
            # Parameter Analysis
            f.write("\n\n" + "="*100 + "\n")
            f.write("PARAMETER IMPACT ANALYSIS:\n")
            f.write("-"*100 + "\n")
            
            # Analyze each parameter's impact
            for param in ['breakeven_trigger_percent', 'trail_method', 'pullback_percent', 'psar_enabled', 'rsi_hook_enabled']:
                if param in df.columns:
                    f.write(f"\n{param}:\n")
                    grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                    for idx, row in grouped.iterrows():
                        f.write(f"  {idx}: Avg P&L ${row['mean']:,.2f} (±${row['std']:,.2f}) | {int(row['count'])} tests\n")
            
            # Recommended configuration
            f.write("\n\n" + "="*100 + "\n")
            f.write("RECOMMENDED CONFIGURATION:\n")
            f.write("-"*100 + "\n")
            
            if not df_sorted.empty:
                best = df_sorted.iloc[0]
                
                config_str = f"""
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
"""
                f.write(config_str)
        
        logging.info(f"📋 Summary saved to {summary_file}")
    
    async def run_optimization(self):
        """Main optimization loop - FIXED VERSION"""
        logging.info("\n" + "="*60)
        logging.info("🔍 DEBUG: run_optimization() called")
        logging.info("="*60)
        
        # Check signals file
        if not self.signals_file.exists():
            logging.error(f"❌ Signals file not found: {self.signals_file}")
            logging.info("Please create: backtester/signals_to_test.txt")
            return
        
        logging.info(f"✅ Found signals file: {self.signals_file}")
        
        # Count signals
        with open(self.signals_file, 'r') as f:
            signal_lines = [line for line in f if line.strip() and not line.startswith('#')]
        
        logging.info(f"📊 Loaded {len(signal_lines)} signals from file")
        
        logging.info(f"\n🚀 Starting parameter optimization")
        logging.info(f"Mode: {'QUICK' if self.quick_mode else 'FULL'}")
        
        # Generate combinations
        logging.info("\n🔍 DEBUG: About to generate combinations...")
        combinations = self.generate_combinations()
        
        if not combinations:
            logging.error("❌ FATAL: No parameter combinations generated!")
            logging.error("🔍 DEBUG: Check get_quick_grid() or get_full_grid() methods")
            return
        
        total_tests = len(combinations)
        
        logging.info(f"\n✅ Generated {total_tests} parameter combinations")
        
        # Estimate time
        time_per_test = 5  # seconds (conservative estimate)
        total_time = (total_tests * time_per_test) / 60
        logging.info(f"⏱️ Estimated time: ~{total_time:.0f} minutes\n")
        
        # Main optimization loop
        logging.info("🔍 DEBUG: Starting main optimization loop...")
        logging.info("="*60)
        
        for i, params in enumerate(combinations, 1):
            logging.info(f"\n🔍 DEBUG: Processing combination {i}/{total_tests}")
            
            if i == 1:
                logging.info(f"🔍 DEBUG: First params: {params}")
            
            result = await self.run_single_test(i, params, total_tests)
            
            if result:
                logging.debug(f"🔍 DEBUG: Test {i} completed successfully")
            else:
                logging.warning(f"🔍 DEBUG: Test {i} returned no result")
            
            # Progress update every 10 tests
            if i % 10 == 0:
                logging.info(f"\n📊 PROGRESS: {i}/{total_tests} tests complete ({i/total_tests*100:.1f}%)")
                if self.results:
                    best_so_far = max(self.results, key=lambda x: x['total_pnl'])
                    logging.info(f"   Best P&L so far: ${best_so_far['total_pnl']:,.2f}")
        
        logging.info("\n🔍 DEBUG: All tests completed, generating report...")
        
        # Generate report
        self.generate_report()
        
        logging.info(f"\n✅ Optimization complete!")
        logging.info(f"📂 Results saved in: {self.output_dir}")
        
        if self.results:
            best_result = max(self.results, key=lambda x: x['total_pnl'])
            logging.info(f"🏆 Best configuration achieved P&L: ${best_result['total_pnl']:,.2f}")
        
        logging.info("🔍 DEBUG: run_optimization() completed")


async def main():
    """Main entry point with command-line interface"""
    import argparse
    
    logging.info("🔍 DEBUG: main() function started")
    
    parser = argparse.ArgumentParser(
        description="Parameter optimization for options backtesting"
    )
    parser.add_argument(
        '--quick', 
        action='store_true', 
        help='Use quick mode (16 tests instead of full grid)'
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
    
    logging.info(f"🔍 DEBUG: Command line args: {args}")
    
    # Create optimizer
    logging.info("🔍 DEBUG: Creating ParameterOptimizer...")
    optimizer = ParameterOptimizer(
        signals_file=args.signals,
        quick_mode=args.quick
    )
    
    # Load custom params if provided
    if args.params:
        logging.info(f"🔍 DEBUG: Loading custom parameters from {args.params}")
        with open(args.params, 'r') as f:
            optimizer.param_grid = json.load(f)
        logging.info(f"Loaded custom parameter grid from {args.params}")
    
    # Run optimization
    logging.info("🔍 DEBUG: Starting optimization...")
    await optimizer.run_optimization()
    
    logging.info("🔍 DEBUG: main() function completed")


if __name__ == "__main__":
    logging.info("\n" + "="*60)
    logging.info("🔍 DEBUG: Script started directly")
    logging.info("="*60)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}", exc_info=True)
    
    logging.info("\n🔍 DEBUG: Script ended")