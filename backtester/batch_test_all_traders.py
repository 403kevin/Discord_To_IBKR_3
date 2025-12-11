#!/usr/bin/env python3
"""
batch_test_all_traders.py - COMPREHENSIVE VERSION v2
=====================================================
Tests ALL traders with THREE modes:
- quick: Fast validation (~200 combinations, ~20 min total)
- balanced: Diverse exploration (~3,000 combinations, ~3-4 hours total)  
- full: Maximum coverage (~15,000+ combinations, ~15-20 hours total)

OUTPUT INCLUDES:
- Per-trader results with best configurations
- Fine-tune recommendations based on top performers
- Master comparison across all traders

Author: 403-Forbidden
Updated: 2025-12-11
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Optional
import json
import argparse

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtester.backtest_engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ComprehensiveBatchTester:
    """
    Comprehensive batch tester for all trading channels.
    Tests every parameter combination without shortcuts.
    """
    
    # All traders to test
    ALL_TRADERS = [
        'goldman', 'qiqo', 'gandalf', 'zeus', 'waxui', 
        'nitro', 'money_mo', 'diesel', 'prophet', 'expo'
    ]
    
    def __init__(self, traders: List[str] = None, mode: str = 'balanced'):
        """
        Initialize batch tester.
        
        Args:
            traders: List of trader names to test (default: all)
            mode: 'quick' (validation), 'balanced' (exploration), 'full' (comprehensive)
        """
        self.traders = traders or self.ALL_TRADERS
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_output_dir = Path(f"backtester/optimization_results/batch_{self.timestamp}")
        self.master_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Results storage
        self.all_trader_results = {}
        self.master_comparison = []
        
        # Get parameter grid based on mode
        self.param_grid = self._get_param_grid()
        self.native_trail_only_grid = self._get_native_trail_only_grid()
        
        # Calculate total combinations
        self.total_combinations = self._count_combinations(self.param_grid)
        self.native_only_combinations = self._count_combinations(self.native_trail_only_grid)
        
        logging.info("=" * 100)
        logging.info("COMPREHENSIVE BATCH TESTER INITIALIZED")
        logging.info("=" * 100)
        logging.info(f"Mode: {mode.upper()}")
        logging.info(f"Traders to test: {len(self.traders)}")
        logging.info(f"Parameter combinations per trader: {self.total_combinations:,}")
        logging.info(f"Native-trail-only combinations: {self.native_only_combinations}")
        logging.info(f"Total tests: {(self.total_combinations + self.native_only_combinations) * len(self.traders):,}")
        logging.info(f"Output directory: {self.master_output_dir}")
        logging.info("=" * 100)
    
    def _get_param_grid(self) -> Dict:
        """Get parameter grid based on mode."""
        if self.mode == 'quick':
            return self._get_quick_grid()
        elif self.mode == 'balanced':
            return self._get_balanced_grid()
        else:  # full
            return self._get_full_grid()
    
    def _get_quick_grid(self) -> Dict:
        """
        QUICK MODE - Fast validation (~200 combinations)
        Purpose: Verify everything runs, get rough directional signal
        Time: ~2 min per trader, ~20 min total
        """
        return {
            # Breakeven - 4 key values
            'breakeven_trigger_percent': [5, 10, 15, 20],
            
            # Trail method - both
            'trail_method': ['pullback_percent', 'atr'],
            
            # Pullback - 4 values including wider for 0DTE
            'pullback_percent': [8, 10, 15, 20],
            
            # ATR - simplified
            'atr_period': [14],
            'atr_multiplier': [1.5, 2.0],
            
            # Native trail - 3 key values
            'native_trail_percent': [20, 25, 30],
            
            # PSAR - disabled for quick
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            
            # RSI - disabled for quick
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30],
        }
    
    def _get_balanced_grid(self) -> Dict:
        """
        BALANCED MODE - Diverse exploration (~3,000 combinations)
        Purpose: Find optimal parameter ranges with good coverage
        Time: ~20-30 min per trader, ~3-4 hours total
        """
        return {
            # ================================================================
            # BREAKEVEN - When to lock in entry price as stop
            # Wider range to find sweet spot
            # ================================================================
            'breakeven_trigger_percent': [3, 5, 7, 10, 12, 15, 20, 25],  # 8 values
            
            # ================================================================
            # TRAIL METHOD - How to trail the stop
            # ================================================================
            'trail_method': ['pullback_percent', 'atr'],  # 2 values
            
            # ================================================================
            # PULLBACK - Fixed % trailing from peak
            # Wide range for both tight scalps and 0DTE breathing room
            # ================================================================
            'pullback_percent': [5, 7, 10, 12, 15, 20, 25],  # 7 values
            
            # ================================================================
            # ATR - Volatility-adjusted trailing
            # ================================================================
            'atr_period': [5, 10, 14, 20, 30],           # 5 values
            'atr_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # 6 values
            
            # ================================================================
            # NATIVE TRAIL - Broker-level safety net (airbag)
            # ================================================================
            'native_trail_percent': [15, 20, 25, 30, 35, 40],  # 6 values
            
            # ================================================================
            # PSAR - Test on/off only (detailed PSAR tuning in full mode)
            # ================================================================
            'psar_enabled': [True, False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            
            # ================================================================
            # RSI - Test on/off only (detailed RSI tuning in full mode)
            # ================================================================
            'rsi_hook_enabled': [True, False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30],
        }
    
    def _get_full_grid(self) -> Dict:
        """
        FULL MODE - Maximum parameter exploration (~15,000+ combinations)
        Purpose: Leave no stone unturned, find global optimum
        Time: ~1.5-2 hours per trader, ~15-20 hours total
        """
        return {
            # ================================================================
            # BREAKEVEN - Extensive range
            # ================================================================
            'breakeven_trigger_percent': [3, 5, 7, 10, 12, 15, 20, 25, 30],  # 9 values
            
            # ================================================================
            # TRAIL METHOD
            # ================================================================
            'trail_method': ['pullback_percent', 'atr'],  # 2 values
            
            # ================================================================
            # PULLBACK - Full spectrum
            # ================================================================
            'pullback_percent': [5, 7, 8, 10, 12, 15, 18, 20, 25, 30],  # 10 values
            
            # ================================================================
            # ATR - Comprehensive
            # ================================================================
            'atr_period': [5, 7, 10, 14, 20, 30],        # 6 values
            'atr_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],  # 7 values
            
            # ================================================================
            # NATIVE TRAIL - Full range including extremes
            # ================================================================
            'native_trail_percent': [10, 15, 20, 25, 30, 35, 40, 50],  # 8 values
            
            # ================================================================
            # PSAR - Full parameter exploration
            # ================================================================
            'psar_enabled': [True, False],
            'psar_start': [0.01, 0.02, 0.03],
            'psar_increment': [0.01, 0.02, 0.03],
            'psar_max': [0.1, 0.2, 0.3],
            
            # ================================================================
            # RSI - Full parameter exploration
            # ================================================================
            'rsi_hook_enabled': [True, False],
            'rsi_period': [7, 10, 14, 20],
            'rsi_overbought': [65, 70, 75, 80],
            'rsi_oversold': [20, 25, 30, 35],
        }
    
    def _get_native_trail_only_grid(self) -> Dict:
        """
        Native trail ONLY grid - tests pure broker trailing stop.
        Disables all software exits to see baseline performance.
        """
        return {
            'breakeven_trigger_percent': [999],  # Never triggers
            'trail_method': ['pullback_percent'],
            'pullback_percent': [999],           # Never triggers
            'atr_period': [14],
            'atr_multiplier': [1.5],
            'native_trail_percent': [8, 10, 12, 15, 18, 20, 25, 30, 35, 40],  # 10 values
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30],
        }
    
    def _count_combinations(self, grid: Dict) -> int:
        """Count total parameter combinations."""
        count = 1
        for key, values in grid.items():
            count *= len(values)
        return count
    
    def _generate_param_combinations(self, grid: Dict) -> List[Dict]:
        """Generate all parameter combinations from grid."""
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        
        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    async def test_single_trader(self, trader_name: str) -> Dict:
        """
        Test a single trader with full parameter optimization.
        """
        logging.info("\n" + "=" * 100)
        logging.info(f"TESTING: {trader_name.upper()}")
        logging.info("=" * 100)
        
        # Find signals file
        signals_file = self._find_signals_file(trader_name)
        
        if not signals_file:
            logging.warning(f"No signals file found for {trader_name}")
            return self._empty_trader_result(trader_name, "No signals file found")
        
        logging.info(f"Signals file: {signals_file}")
        
        # Count signals
        signal_count = self._count_signals(signals_file)
        logging.info(f"Signal count: {signal_count}")
        
        if signal_count < 5:
            logging.warning(f"Too few signals ({signal_count}) for meaningful optimization")
            return self._empty_trader_result(trader_name, f"Too few signals ({signal_count})")
        
        # Create trader output directory
        trader_output_dir = self.master_output_dir / trader_name
        trader_output_dir.mkdir(exist_ok=True)
        
        # Run main optimization
        logging.info(f"\nRunning {self.total_combinations:,} parameter combinations...")
        main_results = await self._run_optimization(
            signals_file, 
            self.param_grid, 
            "main",
            trader_output_dir
        )
        
        # Run native-trail-only optimization
        logging.info(f"\nRunning {self.native_only_combinations} native-trail-only tests...")
        native_results = await self._run_optimization(
            signals_file,
            self.native_trail_only_grid,
            "native_only",
            trader_output_dir
        )
        
        # Analyze and generate report
        analysis = self._analyze_results(trader_name, main_results, native_results, trader_output_dir)
        
        # Store results
        self.all_trader_results[trader_name] = analysis
        self.master_comparison.append({
            'trader': trader_name,
            'signal_count': signal_count,
            'best_pnl': analysis['best_config']['total_pnl'] if analysis['best_config'] else 0,
            'best_win_rate': analysis['best_config']['win_rate'] if analysis['best_config'] else 0,
            'best_profit_factor': analysis['best_config']['profit_factor'] if analysis['best_config'] else 0,
            'profitable_configs': analysis['profitable_configs'],
            'profitable_pct': analysis['profitable_pct'],
            'recommendation': analysis['recommendation']
        })
        
        return analysis
    
    def _find_signals_file(self, trader_name: str) -> Optional[Path]:
        """Find the signals file for a trader."""
        possible_paths = [
            Path(f"backtester/channel_signals/{trader_name}_signals.txt"),
            Path(f"backtester/channel_signals/{trader_name.lower()}_signals.txt"),
            Path(f"backtester/channel_signals/{trader_name.upper()}_signals.txt"),
            Path(f"backtester/{trader_name}_signals.txt"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _count_signals(self, signals_file: Path) -> int:
        """Count valid signals in file."""
        count = 0
        with open(signals_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Trader:'):
                    count += 1
        return count
    
    async def _run_optimization(
        self, 
        signals_file: Path, 
        param_grid: Dict, 
        test_type: str,
        output_dir: Path
    ) -> List[Dict]:
        """Run optimization with given parameter grid."""
        combinations = self._generate_param_combinations(param_grid)
        total = len(combinations)
        results = []
        
        start_time = datetime.now()
        
        for i, params in enumerate(combinations, 1):
            # Progress logging every 100 tests or at key milestones
            if i % 100 == 0 or i == 1 or i == total:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                eta_seconds = (total - i) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logging.info(f"  [{i:,}/{total:,}] ({i/total*100:.1f}%) - {rate:.1f} tests/sec - ETA: {eta_minutes:.1f} min")
            
            try:
                engine = BacktestEngine(
                    signal_file_path=str(signals_file),
                    data_folder_path="backtester/historical_data"
                )
                
                result = engine.run_simulation(params)
                
                if result and result.get('total_trades', 0) > 0:
                    result_entry = {
                        'test_type': test_type,
                        'total_trades': result['total_trades'],
                        'total_pnl': result['total_pnl'],
                        'win_rate': result['win_rate'],
                        'avg_win': result.get('avg_win', 0),
                        'avg_loss': result.get('avg_loss', 0),
                        'profit_factor': result.get('profit_factor', 0),
                        'max_drawdown': result.get('max_drawdown', 0),
                        'final_capital': result.get('final_capital', 10000),
                        'return_pct': result.get('return_pct', 0),
                        'avg_minutes_held': result.get('avg_minutes_held', 0),
                        **params
                    }
                    results.append(result_entry)
                    
            except Exception as e:
                logging.debug(f"  Error in test {i}: {str(e)}")
                continue
        
        # Save raw results to CSV
        if results:
            df = pd.DataFrame(results)
            csv_path = output_dir / f"{test_type}_all_results.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logging.info(f"  Saved {len(results)} results to {csv_path}")
        
        return results
    
    def _analyze_results(
        self, 
        trader_name: str, 
        main_results: List[Dict], 
        native_results: List[Dict],
        output_dir: Path
    ) -> Dict:
        """Analyze results and generate recommendations."""
        
        if not main_results:
            return self._empty_trader_result(trader_name, "No valid results")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(main_results)
        
        # Find profitable configurations
        profitable = df[df['total_pnl'] > 0]
        profitable_count = len(profitable)
        profitable_pct = (profitable_count / len(df)) * 100 if len(df) > 0 else 0
        
        # Find best configuration by P&L
        best_idx = df['total_pnl'].idxmax()
        best_config = df.loc[best_idx].to_dict()
        
        # Find top 10 configurations
        top_10 = df.nlargest(10, 'total_pnl')
        
        # Find best by profit factor (minimum 3 trades)
        valid_pf = df[df['total_trades'] >= 3]
        best_pf_config = None
        if len(valid_pf) > 0:
            best_pf_idx = valid_pf['profit_factor'].idxmax()
            best_pf_config = valid_pf.loc[best_pf_idx].to_dict()
        
        # Find best by win rate (minimum 5 trades)
        valid_wr = df[df['total_trades'] >= 5]
        best_wr_config = None
        if len(valid_wr) > 0:
            best_wr_idx = valid_wr['win_rate'].idxmax()
            best_wr_config = valid_wr.loc[best_wr_idx].to_dict()
        
        # Analyze native-trail-only results
        native_best = None
        if native_results:
            native_df = pd.DataFrame(native_results)
            if len(native_df) > 0:
                native_best_idx = native_df['total_pnl'].idxmax()
                native_best = native_df.loc[native_best_idx].to_dict()
        
        # Generate fine-tune recommendations
        fine_tune_grid = self._generate_fine_tune_recommendations(df, best_config)
        
        # Determine overall recommendation
        recommendation = self._generate_recommendation(
            trader_name, best_config, profitable_pct, fine_tune_grid
        )
        
        # Save analysis report
        self._save_trader_report(
            trader_name, output_dir, df, best_config, top_10, 
            profitable_count, profitable_pct, native_best, fine_tune_grid, recommendation
        )
        
        return {
            'trader_name': trader_name,
            'total_tests': len(df),
            'profitable_configs': profitable_count,
            'profitable_pct': profitable_pct,
            'best_config': best_config,
            'best_pf_config': best_pf_config,
            'best_wr_config': best_wr_config,
            'top_10': top_10.to_dict('records'),
            'native_best': native_best,
            'fine_tune_grid': fine_tune_grid,
            'recommendation': recommendation
        }
    
    def _generate_fine_tune_recommendations(self, df: pd.DataFrame, best_config: Dict) -> Dict:
        """
        Generate fine-tune parameter grid based on top performers.
        Analyzes patterns in profitable configurations to suggest focused ranges.
        """
        # Get top 20% of configurations by P&L
        top_pct = max(1, int(len(df) * 0.20))
        top_configs = df.nlargest(top_pct, 'total_pnl')
        
        # Only consider profitable configs for fine-tuning
        profitable_top = top_configs[top_configs['total_pnl'] > 0]
        
        if len(profitable_top) < 3:
            # Not enough profitable configs - use best config as center
            return self._center_grid_around_config(best_config)
        
        fine_tune = {}
        
        # Analyze each parameter's distribution in top performers
        param_columns = [
            'breakeven_trigger_percent', 'pullback_percent', 
            'atr_period', 'atr_multiplier', 'native_trail_percent'
        ]
        
        for param in param_columns:
            if param in profitable_top.columns:
                values = profitable_top[param].values
                
                # Get range that captures top performers
                p25 = np.percentile(values, 25)
                p75 = np.percentile(values, 75)
                median = np.median(values)
                
                # Create focused range around the sweet spot
                if param in ['breakeven_trigger_percent', 'pullback_percent', 'native_trail_percent']:
                    # Integer percentages
                    low = max(1, int(p25 - 2))
                    high = int(p75 + 2)
                    step = max(1, (high - low) // 5)
                    fine_tune[param] = list(range(low, high + 1, step))
                elif param == 'atr_period':
                    # ATR period
                    low = max(3, int(p25 - 2))
                    high = int(p75 + 2)
                    fine_tune[param] = [low, int(median), high]
                elif param == 'atr_multiplier':
                    # ATR multiplier (floats)
                    low = max(0.5, round(p25 - 0.25, 1))
                    high = round(p75 + 0.25, 1)
                    mid = round(median, 1)
                    fine_tune[param] = sorted(list(set([low, mid, high])))
        
        # Trail method - use whichever is more common in top performers
        if 'trail_method' in profitable_top.columns:
            method_counts = profitable_top['trail_method'].value_counts()
            if len(method_counts) > 0:
                best_method = method_counts.index[0]
                fine_tune['trail_method'] = [best_method]
        
        # PSAR/RSI - check if enabled configs perform better
        for indicator in ['psar_enabled', 'rsi_hook_enabled']:
            if indicator in profitable_top.columns:
                enabled_pnl = profitable_top[profitable_top[indicator] == True]['total_pnl'].mean() if True in profitable_top[indicator].values else 0
                disabled_pnl = profitable_top[profitable_top[indicator] == False]['total_pnl'].mean() if False in profitable_top[indicator].values else 0
                fine_tune[indicator] = [enabled_pnl > disabled_pnl]
        
        return fine_tune
    
    def _center_grid_around_config(self, config: Dict) -> Dict:
        """Create a fine-tune grid centered around a single best config."""
        fine_tune = {}
        
        be = config.get('breakeven_trigger_percent', 10)
        fine_tune['breakeven_trigger_percent'] = [max(1, be-3), be, be+3]
        
        pb = config.get('pullback_percent', 10)
        fine_tune['pullback_percent'] = [max(3, pb-3), pb, pb+3]
        
        atr_p = config.get('atr_period', 14)
        fine_tune['atr_period'] = [max(5, atr_p-3), atr_p, atr_p+3]
        
        atr_m = config.get('atr_multiplier', 1.5)
        fine_tune['atr_multiplier'] = [max(0.5, round(atr_m-0.5, 1)), atr_m, round(atr_m+0.5, 1)]
        
        nt = config.get('native_trail_percent', 25)
        fine_tune['native_trail_percent'] = [max(10, nt-5), nt, nt+5]
        
        fine_tune['trail_method'] = [config.get('trail_method', 'pullback_percent')]
        fine_tune['psar_enabled'] = [config.get('psar_enabled', False)]
        fine_tune['rsi_hook_enabled'] = [config.get('rsi_hook_enabled', False)]
        
        return fine_tune
    
    def _generate_recommendation(
        self, 
        trader_name: str, 
        best_config: Dict, 
        profitable_pct: float,
        fine_tune_grid: Dict
    ) -> str:
        """Generate overall recommendation for trader."""
        
        pnl = best_config.get('total_pnl', 0)
        win_rate = best_config.get('win_rate', 0)
        profit_factor = best_config.get('profit_factor', 0)
        trades = best_config.get('total_trades', 0)
        
        # Scoring
        score = 0
        reasons = []
        
        if pnl > 500:
            score += 3
            reasons.append(f"Strong P&L (${pnl:.0f})")
        elif pnl > 100:
            score += 2
            reasons.append(f"Positive P&L (${pnl:.0f})")
        elif pnl > 0:
            score += 1
            reasons.append(f"Marginal P&L (${pnl:.0f})")
        else:
            reasons.append(f"Negative P&L (${pnl:.0f})")
        
        if win_rate > 60:
            score += 2
            reasons.append(f"High win rate ({win_rate:.1f}%)")
        elif win_rate > 50:
            score += 1
            reasons.append(f"Decent win rate ({win_rate:.1f}%)")
        
        if profit_factor > 2.0:
            score += 2
            reasons.append(f"Excellent profit factor ({profit_factor:.2f})")
        elif profit_factor > 1.5:
            score += 1
            reasons.append(f"Good profit factor ({profit_factor:.2f})")
        
        if profitable_pct > 20:
            score += 1
            reasons.append(f"Many profitable configs ({profitable_pct:.1f}%)")
        
        if trades < 10:
            score -= 1
            reasons.append(f"Low sample size ({trades} trades)")
        
        # Generate recommendation
        if score >= 6:
            verdict = "🟢 HIGHLY RECOMMENDED - Strong performer, proceed to fine-tuning"
        elif score >= 4:
            verdict = "🟡 PROMISING - Worth fine-tuning, monitor closely"
        elif score >= 2:
            verdict = "🟠 MARGINAL - Consider with caution, needs more data"
        else:
            verdict = "🔴 NOT RECOMMENDED - Poor performance, skip or re-evaluate"
        
        return f"{verdict}\n   Reasons: {', '.join(reasons)}"
    
    def _save_trader_report(
        self, 
        trader_name: str, 
        output_dir: Path, 
        df: pd.DataFrame,
        best_config: Dict,
        top_10: pd.DataFrame,
        profitable_count: int,
        profitable_pct: float,
        native_best: Dict,
        fine_tune_grid: Dict,
        recommendation: str
    ):
        """Save detailed trader report."""
        
        report_path = output_dir / f"{trader_name}_optimization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"OPTIMIZATION REPORT: {trader_name.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total configurations tested: {len(df):,}\n")
            f.write(f"Profitable configurations: {profitable_count:,} ({profitable_pct:.1f}%)\n")
            f.write(f"Best P&L: ${best_config.get('total_pnl', 0):,.2f}\n")
            f.write(f"Best Win Rate: {best_config.get('win_rate', 0):.1f}%\n")
            f.write(f"Best Profit Factor: {best_config.get('profit_factor', 0):.2f}\n\n")
            
            # Recommendation
            f.write("RECOMMENDATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"{recommendation}\n\n")
            
            # Best Configuration
            f.write("BEST CONFIGURATION (by P&L)\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Breakeven Trigger: {best_config.get('breakeven_trigger_percent')}%\n")
            f.write(f"  Trail Method: {best_config.get('trail_method')}\n")
            f.write(f"  Pullback Percent: {best_config.get('pullback_percent')}%\n")
            f.write(f"  ATR Period: {best_config.get('atr_period')}\n")
            f.write(f"  ATR Multiplier: {best_config.get('atr_multiplier')}\n")
            f.write(f"  Native Trail: {best_config.get('native_trail_percent')}%\n")
            f.write(f"  PSAR Enabled: {best_config.get('psar_enabled')}\n")
            f.write(f"  RSI Hook Enabled: {best_config.get('rsi_hook_enabled')}\n")
            f.write(f"\n  Results:\n")
            f.write(f"    Total Trades: {best_config.get('total_trades')}\n")
            f.write(f"    Total P&L: ${best_config.get('total_pnl', 0):,.2f}\n")
            f.write(f"    Win Rate: {best_config.get('win_rate', 0):.1f}%\n")
            f.write(f"    Profit Factor: {best_config.get('profit_factor', 0):.2f}\n")
            f.write(f"    Avg Hold Time: {best_config.get('avg_minutes_held', 0):.0f} min\n\n")
            
            # Top 10
            f.write("TOP 10 CONFIGURATIONS\n")
            f.write("-" * 40 + "\n")
            for i, row in top_10.iterrows():
                f.write(f"  #{top_10.index.get_loc(i)+1}: P&L=${row['total_pnl']:.0f} | "
                       f"WR={row['win_rate']:.0f}% | "
                       f"BE={row['breakeven_trigger_percent']}% | "
                       f"Trail={row['trail_method']} | "
                       f"PB={row['pullback_percent']}% | "
                       f"NT={row['native_trail_percent']}%\n")
            f.write("\n")
            
            # Native Trail Only Results
            if native_best:
                f.write("NATIVE TRAIL ONLY (Baseline)\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Best Native Trail: {native_best.get('native_trail_percent')}%\n")
                f.write(f"  P&L: ${native_best.get('total_pnl', 0):,.2f}\n")
                f.write(f"  Win Rate: {native_best.get('win_rate', 0):.1f}%\n\n")
            
            # Fine-Tune Recommendations
            f.write("FINE-TUNE RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Based on top performers, focus testing on these ranges:\n\n")
            f.write("```json\n")
            f.write(json.dumps(fine_tune_grid, indent=2))
            f.write("\n```\n\n")
            
            # Config.py snippet
            f.write("CONFIG.PY SNIPPET (Best Config)\n")
            f.write("-" * 40 + "\n")
            f.write(f'''"exit_strategy": {{
    "breakeven_trigger_percent": {best_config.get('breakeven_trigger_percent')},
    "trail_method": "{best_config.get('trail_method')}",
    "trail_settings": {{
        "pullback_percent": {best_config.get('pullback_percent')},
        "atr_period": {best_config.get('atr_period')},
        "atr_multiplier": {best_config.get('atr_multiplier')}
    }},
    "momentum_exits": {{
        "psar_enabled": {str(best_config.get('psar_enabled')).lower()},
        "rsi_hook_enabled": {str(best_config.get('rsi_hook_enabled')).lower()}
    }}
}},
"safety_net": {{"enabled": True, "native_trail_percent": {best_config.get('native_trail_percent')}}}
''')
            f.write("\n")
        
        logging.info(f"  Saved report to {report_path}")
    
    def _empty_trader_result(self, trader_name: str, reason: str) -> Dict:
        """Return empty result structure for failed traders."""
        return {
            'trader_name': trader_name,
            'total_tests': 0,
            'profitable_configs': 0,
            'profitable_pct': 0,
            'best_config': None,
            'best_pf_config': None,
            'best_wr_config': None,
            'top_10': [],
            'native_best': None,
            'fine_tune_grid': {},
            'recommendation': f"⚠️ SKIPPED - {reason}"
        }
    
    async def run_all_traders(self):
        """Run optimization for all traders."""
        logging.info("\n" + "=" * 100)
        logging.info("STARTING BATCH OPTIMIZATION")
        logging.info(f"Testing {len(self.traders)} traders in {self.mode.upper()} mode")
        logging.info("=" * 100)
        
        start_time = datetime.now()
        
        for i, trader in enumerate(self.traders, 1):
            logging.info(f"\n[{i}/{len(self.traders)}] Processing {trader}...")
            await self.test_single_trader(trader)
        
        elapsed = datetime.now() - start_time
        
        # Generate master report
        self._generate_master_report(elapsed)
        
        logging.info("\n" + "=" * 100)
        logging.info("BATCH OPTIMIZATION COMPLETE")
        logging.info(f"Total time: {elapsed}")
        logging.info(f"Results saved to: {self.master_output_dir}")
        logging.info("=" * 100)
    
    def _generate_master_report(self, elapsed):
        """Generate master comparison report across all traders."""
        
        report_path = self.master_output_dir / "MASTER_COMPARISON.txt"
        
        # Sort by best P&L
        sorted_comparison = sorted(self.master_comparison, key=lambda x: x['best_pnl'], reverse=True)
        
        with open(report_path, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MASTER COMPARISON REPORT - ALL TRADERS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode.upper()}\n")
            f.write(f"Total Time: {elapsed}\n")
            f.write("=" * 100 + "\n\n")
            
            # Rankings
            f.write("TRADER RANKINGS (by Best P&L)\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6}{'Trader':<15}{'Best P&L':<12}{'Win Rate':<10}{'PF':<8}{'Profitable %':<14}{'Signals':<10}\n")
            f.write("-" * 80 + "\n")
            
            for i, trader in enumerate(sorted_comparison, 1):
                f.write(f"{i:<6}{trader['trader']:<15}"
                       f"${trader['best_pnl']:>8,.0f}   "
                       f"{trader['best_win_rate']:>6.1f}%   "
                       f"{trader['best_profit_factor']:>5.2f}   "
                       f"{trader['profitable_pct']:>10.1f}%   "
                       f"{trader['signal_count']:>6}\n")
            
            f.write("\n")
            
            # Recommendations Summary
            f.write("RECOMMENDATIONS SUMMARY\n")
            f.write("-" * 80 + "\n")
            for trader in sorted_comparison:
                f.write(f"\n{trader['trader'].upper()}:\n")
                f.write(f"  {trader['recommendation']}\n")
            
            f.write("\n")
            
            # Top 3 Traders to Focus On
            f.write("=" * 80 + "\n")
            f.write("TOP 3 TRADERS TO FOCUS ON\n")
            f.write("=" * 80 + "\n")
            
            profitable_traders = [t for t in sorted_comparison if t['best_pnl'] > 0]
            for i, trader in enumerate(profitable_traders[:3], 1):
                f.write(f"\n{i}. {trader['trader'].upper()}\n")
                f.write(f"   Best P&L: ${trader['best_pnl']:,.0f}\n")
                f.write(f"   Win Rate: {trader['best_win_rate']:.1f}%\n")
                f.write(f"   See: {trader['trader']}/{trader['trader']}_optimization_report.txt\n")
            
            if not profitable_traders:
                f.write("\n⚠️ No profitable traders found in this test run.\n")
                f.write("Consider: More signals, different parameters, or different traders.\n")
        
        # Also save as CSV for easy analysis
        csv_path = self.master_output_dir / "master_comparison.csv"
        pd.DataFrame(sorted_comparison).to_csv(csv_path, index=False)
        
        logging.info(f"Master report saved to {report_path}")


async def main():
    parser = argparse.ArgumentParser(description='Comprehensive batch tester for all traders')
    parser.add_argument('--mode', choices=['quick', 'balanced', 'full'], default='balanced',
                       help='Testing mode: quick (~20min), balanced (~3-4hrs), full (~15-20hrs)')
    parser.add_argument('--traders', nargs='+', help='Specific traders to test (default: all)')
    
    args = parser.parse_args()
    
    tester = ComprehensiveBatchTester(
        traders=args.traders,
        mode=args.mode
    )
    
    await tester.run_all_traders()


if __name__ == "__main__":
    asyncio.run(main())
