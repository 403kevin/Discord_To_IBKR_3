#!/usr/bin/env python3
"""
batch_test_all_traders.py - COMPREHENSIVE VERSION
==================================================
Tests ALL traders with EXPANDED parameter grid.
No shortcuts - thorough testing of every variable combination.

PARAMETER COUNT:
- Breakeven triggers: 8 values
- Trail methods: 2 values
- Pullback percentages: 8 values
- ATR periods: 6 values
- ATR multipliers: 7 values
- Native trail percentages: 7 values
- PSAR enabled: 2 values
- PSAR start: 3 values
- PSAR increment: 3 values
- PSAR max: 3 values
- RSI enabled: 2 values
- RSI period: 4 values
- RSI overbought: 4 values
- RSI oversold: 4 values

ESTIMATED COMBINATIONS: ~15,000+ per trader (depends on grid)
ESTIMATED TIME: 4-8 hours per trader

Author: 403-Forbidden
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
    
    def __init__(self, traders: List[str] = None, mode: str = 'full'):
        """
        Initialize batch tester.
        
        Args:
            traders: List of trader names to test (default: all)
            mode: 'full' (comprehensive), 'standard' (balanced), 'quick' (fast screening)
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
        elif self.mode == 'standard':
            return self._get_standard_grid()
        else:  # full
            return self._get_full_grid()
    
    def _get_quick_grid(self) -> Dict:
        """Quick screening grid - ~100 combinations"""
        return {
            # Breakeven
            'breakeven_trigger_percent': [5, 7, 10, 15],
            
            # Trail method
            'trail_method': ['pullback_percent', 'atr'],
            
            # Pullback
            'pullback_percent': [8, 10, 15],
            
            # ATR
            'atr_period': [14],
            'atr_multiplier': [1.5, 2.0],
            
            # Native trail
            'native_trail_percent': [20, 25, 30],
            
            # PSAR (disabled for quick)
            'psar_enabled': [False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            
            # RSI (disabled for quick)
            'rsi_hook_enabled': [False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30],
        }
    
    def _get_standard_grid(self) -> Dict:
        """Standard grid - ~2,000 combinations"""
        return {
            # Breakeven - more granular
            'breakeven_trigger_percent': [3, 5, 7, 10, 12, 15],
            
            # Trail method
            'trail_method': ['pullback_percent', 'atr'],
            
            # Pullback - expanded
            'pullback_percent': [5, 7, 10, 12, 15, 20],
            
            # ATR - expanded
            'atr_period': [5, 10, 14, 20],
            'atr_multiplier': [1.0, 1.5, 2.0, 2.5],
            
            # Native trail - more options
            'native_trail_percent': [15, 20, 25, 30, 35],
            
            # PSAR - test on/off
            'psar_enabled': [True, False],
            'psar_start': [0.02],
            'psar_increment': [0.02],
            'psar_max': [0.2],
            
            # RSI - test on/off
            'rsi_hook_enabled': [True, False],
            'rsi_period': [14],
            'rsi_overbought': [70],
            'rsi_oversold': [30],
        }
    
    def _get_full_grid(self) -> Dict:
        """
        FULL COMPREHENSIVE GRID - ~15,000+ combinations
        Tests EVERY meaningful parameter variation.
        """
        return {
            # ================================================================
            # BREAKEVEN TRIGGER - When to activate breakeven stop
            # ================================================================
            # Range: 3% to 25% (8 values)
            # Lower = more aggressive protection
            # Higher = more room to breathe
            'breakeven_trigger_percent': [3, 5, 7, 10, 12, 15, 20, 25],
            
            # ================================================================
            # TRAIL METHOD - How to trail the stop
            # ================================================================
            'trail_method': ['pullback_percent', 'atr'],
            
            # ================================================================
            # PULLBACK PERCENT - Fixed % trailing stop
            # ================================================================
            # Range: 5% to 30% (8 values)
            # Lower = tighter stop, more whipsaw
            # Higher = more room, larger drawdowns
            'pullback_percent': [5, 7, 8, 10, 12, 15, 20, 30],
            
            # ================================================================
            # ATR PERIOD - Lookback for volatility calculation
            # ================================================================
            # Range: 5 to 30 bars (6 values)
            # Lower = more reactive to recent volatility
            # Higher = smoother, slower adaptation
            'atr_period': [5, 7, 10, 14, 20, 30],
            
            # ================================================================
            # ATR MULTIPLIER - How many ATRs for stop distance
            # ================================================================
            # Range: 0.5 to 3.5 (7 values)
            # Lower = tighter stop
            # Higher = wider stop
            'atr_multiplier': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            
            # ================================================================
            # NATIVE TRAIL PERCENT - Broker-level safety net
            # ================================================================
            # Range: 10% to 40% (7 values)
            # This is the "airbag" - last resort protection
            'native_trail_percent': [10, 15, 20, 25, 30, 35, 40],
            
            # ================================================================
            # PSAR - Parabolic SAR momentum exit
            # ================================================================
            'psar_enabled': [True, False],
            'psar_start': [0.01, 0.02, 0.03],        # Acceleration start
            'psar_increment': [0.01, 0.02, 0.03],    # Acceleration increment
            'psar_max': [0.1, 0.2, 0.3],             # Max acceleration
            
            # ================================================================
            # RSI HOOK - RSI-based momentum exit
            # ================================================================
            'rsi_hook_enabled': [True, False],
            'rsi_period': [7, 10, 14, 20],           # RSI lookback period
            'rsi_overbought': [65, 70, 75, 80],      # Overbought threshold
            'rsi_oversold': [20, 25, 30, 35],        # Oversold threshold
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
        
        Args:
            trader_name: Name of the trader/channel
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        logging.info("\n" + "=" * 100)
        logging.info(f"[TARGET] TESTING: {trader_name.upper()}")
        logging.info("=" * 100)
        
        # Find signals file
        signals_file = self._find_signals_file(trader_name)
        
        if not signals_file:
            logging.warning(f"[WARNING] No signals file found for {trader_name}")
            return self._empty_trader_result(trader_name, "No signals file found")
        
        logging.info(f"[FOLDER] Signals file: {signals_file}")
        
        # Count signals
        signal_count = self._count_signals(signals_file)
        logging.info(f"[CHART] Signal count: {signal_count}")
        
        if signal_count < 5:
            logging.warning(f"[WARNING] Too few signals ({signal_count}) for meaningful optimization")
            return self._empty_trader_result(trader_name, f"Too few signals ({signal_count})")
        
        # Create trader output directory
        trader_output_dir = self.master_output_dir / trader_name
        trader_output_dir.mkdir(exist_ok=True)
        
        # Run main optimization
        logging.info(f"\n[CYCLE] Running {self.total_combinations:,} parameter combinations...")
        main_results = await self._run_optimization(
            signals_file, 
            self.param_grid, 
            "main",
            trader_output_dir
        )
        
        # Run native-trail-only optimization
        logging.info(f"\n[CYCLE] Running {self.native_only_combinations} native-trail-only tests...")
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
            # Progress logging every 100 tests
            if i % 100 == 0 or i == 1:
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
            logging.info(f"  [SAVE] Saved {len(results)} results to {csv_path}")
        
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
            return self._empty_trader_result(trader_name, "No valid results from backtests")
        
        df_main = pd.DataFrame(main_results)
        df_native = pd.DataFrame(native_results) if native_results else pd.DataFrame()
        
        # Find best configurations
        best_by_pnl = df_main.loc[df_main['total_pnl'].idxmax()].to_dict() if len(df_main) > 0 else None
        best_by_wr = df_main.loc[df_main['win_rate'].idxmax()].to_dict() if len(df_main) > 0 else None
        best_by_pf = df_main.loc[df_main['profit_factor'].idxmax()].to_dict() if len(df_main) > 0 else None
        
        # Count profitable configurations
        profitable_configs = len(df_main[df_main['total_pnl'] > 0])
        total_configs = len(df_main)
        profitable_pct = (profitable_configs / total_configs * 100) if total_configs > 0 else 0
        
        # Best native-only result
        best_native = None
        if len(df_native) > 0:
            best_native = df_native.loc[df_native['total_pnl'].idxmax()].to_dict()
        
        # Parameter sensitivity analysis
        param_analysis = self._analyze_parameter_sensitivity(df_main)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            best_by_pnl, profitable_pct, param_analysis
        )
        
        # Generate detailed report
        self._generate_trader_report(
            trader_name, df_main, df_native, best_by_pnl, best_by_wr, best_by_pf,
            best_native, profitable_configs, param_analysis, recommendation, output_dir
        )
        
        return {
            'trader': trader_name,
            'total_tests': total_configs,
            'profitable_configs': profitable_configs,
            'profitable_pct': profitable_pct,
            'best_config': best_by_pnl,
            'best_by_win_rate': best_by_wr,
            'best_by_profit_factor': best_by_pf,
            'best_native_only': best_native,
            'param_analysis': param_analysis,
            'recommendation': recommendation
        }
    
    def _analyze_parameter_sensitivity(self, df: pd.DataFrame) -> Dict:
        """Analyze which parameters have the biggest impact on P&L."""
        analysis = {}
        
        # Key parameters to analyze
        params_to_check = [
            'breakeven_trigger_percent', 'trail_method', 'pullback_percent',
            'atr_period', 'atr_multiplier', 'native_trail_percent',
            'psar_enabled', 'rsi_hook_enabled'
        ]
        
        for param in params_to_check:
            if param in df.columns:
                try:
                    grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std', 'count'])
                    best_value = grouped['mean'].idxmax()
                    worst_value = grouped['mean'].idxmin()
                    spread = grouped['mean'].max() - grouped['mean'].min()
                    
                    analysis[param] = {
                        'best_value': best_value,
                        'best_avg_pnl': grouped.loc[best_value, 'mean'],
                        'worst_value': worst_value,
                        'worst_avg_pnl': grouped.loc[worst_value, 'mean'],
                        'spread': spread,
                        'importance': 'HIGH' if spread > 100 else 'MEDIUM' if spread > 50 else 'LOW'
                    }
                except Exception:
                    continue
        
        return analysis
    
    def _generate_recommendation(
        self, 
        best_config: Dict, 
        profitable_pct: float,
        param_analysis: Dict
    ) -> str:
        """Generate trading recommendation based on results."""
        
        if not best_config:
            return "[X] AVOID - No profitable configurations found"
        
        pnl = best_config.get('total_pnl', 0)
        wr = best_config.get('win_rate', 0)
        pf = best_config.get('profit_factor', 0)
        
        if pnl < 0:
            return "[X] AVOID - Best configuration is still unprofitable"
        
        if profitable_pct < 10:
            return f"[!] RISKY - Only {profitable_pct:.1f}% of configs profitable (highly parameter-sensitive)"
        
        if pf < 1.2:
            return f"[!] MARGINAL - Low profit factor ({pf:.2f}), edge is thin"
        
        if wr < 40:
            return f"[!] CAUTION - Low win rate ({wr:.1f}%), requires strong risk management"
        
        if profitable_pct > 50 and pf > 1.5 and wr > 50:
            return f"[OK] RECOMMENDED - Robust trader ({profitable_pct:.0f}% configs profitable, PF: {pf:.2f})"
        
        if profitable_pct > 30 and pf > 1.3:
            return f"[+] VIABLE - Decent trader ({profitable_pct:.0f}% configs profitable)"
        
        return f"[?] TEST MORE - Results inconclusive ({profitable_pct:.0f}% profitable)"
    
    def _generate_trader_report(
        self, trader_name: str, df_main: pd.DataFrame, df_native: pd.DataFrame,
        best_pnl: Dict, best_wr: Dict, best_pf: Dict, best_native: Dict,
        profitable_configs: int, param_analysis: Dict, recommendation: str,
        output_dir: Path
    ):
        """Generate comprehensive trader report."""
        
        report = []
        report.append("=" * 100)
        report.append(f"OPTIMIZATION REPORT: {trader_name.upper()}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 100)
        
        # Summary statistics
        report.append("\n" + "-" * 50)
        report.append("SUMMARY STATISTICS")
        report.append("-" * 50)
        report.append(f"Total configurations tested: {len(df_main):,}")
        report.append(f"Profitable configurations: {profitable_configs:,} ({profitable_configs/len(df_main)*100:.1f}%)")
        report.append(f"Average P&L across all configs: ${df_main['total_pnl'].mean():.2f}")
        report.append(f"Median P&L: ${df_main['total_pnl'].median():.2f}")
        report.append(f"P&L Std Dev: ${df_main['total_pnl'].std():.2f}")
        
        # Best by P&L
        report.append("\n" + "-" * 50)
        report.append("BEST CONFIGURATION (by P&L)")
        report.append("-" * 50)
        if best_pnl:
            report.append(f"Total P&L: ${best_pnl['total_pnl']:.2f}")
            report.append(f"Win Rate: {best_pnl['win_rate']:.1f}%")
            report.append(f"Profit Factor: {best_pnl['profit_factor']:.2f}")
            report.append(f"Total Trades: {best_pnl['total_trades']}")
            report.append(f"Avg Hold Time: {best_pnl.get('avg_minutes_held', 0):.0f} minutes")
            report.append("\nParameters:")
            report.append(f"  Breakeven Trigger: {best_pnl['breakeven_trigger_percent']}%")
            report.append(f"  Trail Method: {best_pnl['trail_method']}")
            report.append(f"  Pullback %: {best_pnl['pullback_percent']}%")
            report.append(f"  ATR Period: {best_pnl['atr_period']}")
            report.append(f"  ATR Multiplier: {best_pnl['atr_multiplier']}")
            report.append(f"  Native Trail: {best_pnl['native_trail_percent']}%")
            report.append(f"  PSAR Enabled: {best_pnl['psar_enabled']}")
            report.append(f"  RSI Hook Enabled: {best_pnl['rsi_hook_enabled']}")
        
        # Best by Win Rate
        report.append("\n" + "-" * 50)
        report.append("BEST CONFIGURATION (by Win Rate)")
        report.append("-" * 50)
        if best_wr:
            report.append(f"Win Rate: {best_wr['win_rate']:.1f}%")
            report.append(f"Total P&L: ${best_wr['total_pnl']:.2f}")
            report.append(f"Profit Factor: {best_wr['profit_factor']:.2f}")
        
        # Best by Profit Factor
        report.append("\n" + "-" * 50)
        report.append("BEST CONFIGURATION (by Profit Factor)")
        report.append("-" * 50)
        if best_pf:
            report.append(f"Profit Factor: {best_pf['profit_factor']:.2f}")
            report.append(f"Total P&L: ${best_pf['total_pnl']:.2f}")
            report.append(f"Win Rate: {best_pf['win_rate']:.1f}%")
        
        # Native-trail-only results
        if best_native:
            report.append("\n" + "-" * 50)
            report.append("NATIVE TRAIL ONLY (Baseline)")
            report.append("-" * 50)
            report.append(f"Best Native Trail %: {best_native['native_trail_percent']}%")
            report.append(f"P&L: ${best_native['total_pnl']:.2f}")
            report.append(f"Win Rate: {best_native['win_rate']:.1f}%")
            
            if len(df_native) > 0:
                report.append("\nAll native-trail-only results:")
                for _, row in df_native.sort_values('native_trail_percent').iterrows():
                    report.append(f"  {row['native_trail_percent']:3.0f}% -> ${row['total_pnl']:8.2f} | WR: {row['win_rate']:5.1f}%")
        
        # Parameter sensitivity
        report.append("\n" + "-" * 50)
        report.append("PARAMETER SENSITIVITY ANALYSIS")
        report.append("-" * 50)
        for param, data in param_analysis.items():
            importance = data.get('importance', 'UNKNOWN')
            best_val = data.get('best_value')
            best_pnl_val = data.get('best_avg_pnl', 0)
            worst_val = data.get('worst_value')
            spread = data.get('spread', 0)
            
            report.append(f"\n{param}:")
            report.append(f"  Importance: {importance}")
            report.append(f"  Best value: {best_val} (avg P&L: ${best_pnl_val:.2f})")
            report.append(f"  Worst value: {worst_val}")
            report.append(f"  P&L spread: ${spread:.2f}")
        
        # Top 20 configurations
        report.append("\n" + "-" * 50)
        report.append("TOP 20 CONFIGURATIONS BY P&L")
        report.append("-" * 50)
        top_20 = df_main.nlargest(20, 'total_pnl')
        for i, (_, row) in enumerate(top_20.iterrows(), 1):
            report.append(f"{i:2d}. P&L: ${row['total_pnl']:8.2f} | WR: {row['win_rate']:5.1f}% | "
                         f"PF: {row['profit_factor']:5.2f} | BE: {row['breakeven_trigger_percent']:2.0f}% | "
                         f"Trail: {row['trail_method'][:8]:8s} | Native: {row['native_trail_percent']:2.0f}%")
        
        # Recommendation
        report.append("\n" + "=" * 100)
        report.append("RECOMMENDATION")
        report.append("=" * 100)
        report.append(recommendation)
        
        # Config.py snippet
        if best_pnl and best_pnl.get('total_pnl', 0) > 0:
            report.append("\n" + "-" * 50)
            report.append("COPY-PASTE CONFIG.PY SNIPPET")
            report.append("-" * 50)
            report.append(f'''
"exit_strategy": {{
    "breakeven_trigger_percent": {best_pnl['breakeven_trigger_percent']},
    "exit_priority": ["breakeven", "rsi_hook", "psar_flip", "atr_trail", "pullback_stop"],
    "trail_method": "{best_pnl['trail_method']}",
    "trail_settings": {{
        "pullback_percent": {best_pnl['pullback_percent']},
        "atr_period": {best_pnl['atr_period']},
        "atr_multiplier": {best_pnl['atr_multiplier']}
    }},
    "momentum_exits": {{
        "psar_enabled": {str(best_pnl['psar_enabled'])},
        "psar_settings": {{"start": {best_pnl['psar_start']}, "increment": {best_pnl['psar_increment']}, "max": {best_pnl['psar_max']}}},
        "rsi_hook_enabled": {str(best_pnl['rsi_hook_enabled'])},
        "rsi_settings": {{"period": {best_pnl['rsi_period']}, "overbought_level": {best_pnl['rsi_overbought']}, "oversold_level": {best_pnl['rsi_oversold']}}}
    }}
}},
"safety_net": {{"enabled": True, "native_trail_percent": {best_pnl['native_trail_percent']}}}
''')
        
        # Save report
        report_path = output_dir / f"{trader_name}_optimization_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logging.info(f"  [DOC] Generated report: {report_path}")
    
    def _empty_trader_result(self, trader_name: str, reason: str) -> Dict:
        """Return empty result structure."""
        return {
            'trader': trader_name,
            'total_tests': 0,
            'profitable_configs': 0,
            'profitable_pct': 0,
            'best_config': None,
            'best_by_win_rate': None,
            'best_by_profit_factor': None,
            'best_native_only': None,
            'param_analysis': {},
            'recommendation': f"[X] SKIP - {reason}"
        }
    
    def generate_master_report(self):
        """Generate master comparison report across all traders."""
        
        if not self.master_comparison:
            logging.warning("No results to generate master report")
            return
        
        df = pd.DataFrame(self.master_comparison)
        
        # Sort by best P&L
        df_sorted = df.sort_values('best_pnl', ascending=False)
        
        report = []
        report.append("=" * 100)
        report.append("MASTER COMPARISON REPORT - ALL TRADERS")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Mode: {self.mode.upper()}")
        report.append("=" * 100)
        
        report.append("\n" + "-" * 100)
        report.append("TRADER RANKINGS (by Best P&L)")
        report.append("-" * 100)
        
        for i, row in df_sorted.iterrows():
            status = "[OK]" if row['best_pnl'] > 0 else "[X]"
            report.append(f"{status} {row['trader'].upper():15s} | "
                         f"Best P&L: ${row['best_pnl']:8.2f} | "
                         f"Best WR: {row['best_win_rate']:5.1f}% | "
                         f"Profitable: {row['profitable_configs']:4d} configs | "
                         f"Signals: {row['signal_count']:3d}")
        
        # Profitable traders
        profitable = df_sorted[df_sorted['best_pnl'] > 0]
        unprofitable = df_sorted[df_sorted['best_pnl'] <= 0]
        
        report.append("\n" + "-" * 100)
        report.append("SUMMARY")
        report.append("-" * 100)
        report.append(f"Total traders tested: {len(df)}")
        report.append(f"Profitable traders: {len(profitable)}")
        report.append(f"Unprofitable traders: {len(unprofitable)}")
        
        if len(profitable) > 0:
            report.append(f"\n[OK] RECOMMENDED TRADERS:")
            for _, row in profitable.iterrows():
                report.append(f"   - {row['trader'].upper()}: ${row['best_pnl']:.2f}")
        
        if len(unprofitable) > 0:
            report.append(f"\n[X] AVOID TRADERS:")
            for _, row in unprofitable.iterrows():
                report.append(f"   - {row['trader'].upper()}: ${row['best_pnl']:.2f}")
        
        # Save master report
        report_path = self.master_output_dir / "MASTER_COMPARISON.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Save master CSV
        csv_path = self.master_output_dir / "master_comparison.csv"
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
        
        logging.info(f"\n[CHART] Master report: {report_path}")
        logging.info(f"[CHART] Master CSV: {csv_path}")
    
    async def run(self):
        """Run batch tests on all traders."""
        
        logging.info("\n" + "=" * 100)
        logging.info("STARTING COMPREHENSIVE BATCH TESTING")
        logging.info("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        for i, trader in enumerate(self.traders, 1):
            logging.info(f"\n[{i}/{len(self.traders)}] Testing {trader}...")
            await self.test_single_trader(trader)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate master report
        self.generate_master_report()
        
        logging.info("\n" + "=" * 100)
        logging.info("BATCH TESTING COMPLETE!")
        logging.info(f"Total time: {duration}")
        logging.info(f"Results: {self.master_output_dir}")
        logging.info("=" * 100)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive Batch Tester')
    parser.add_argument('--mode', choices=['quick', 'standard', 'full'], default='full',
                       help='Testing mode: quick (~100 combos), standard (~2000), full (~15000+)')
    parser.add_argument('--traders', nargs='+', default=None,
                       help='Specific traders to test (default: all)')
    
    args = parser.parse_args()
    
    tester = ComprehensiveBatchTester(traders=args.traders, mode=args.mode)
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())
