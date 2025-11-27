#!/usr/bin/env python3
"""
batch_fine_tune.py - COMPREHENSIVE FINE-TUNING
===============================================
Takes the best configurations from batch_test_all_traders.py
and does SURGICAL fine-tuning around those values.

PURPOSE:
- Validate that profitable configs aren't just lucky outliers
- Find the optimal "sweet spot" for each parameter
- Test parameter stability (does +-1-2% change results dramatically?)
- Generate production-ready configurations

WORKFLOW:
1. Load results from batch_test_all_traders.py
2. For each profitable trader, extract best config
3. Create fine-tuning grid +-20% around each optimal parameter
4. Run comprehensive tests
5. Compare to original to detect overfitting

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
from typing import Dict, List, Optional, Tuple
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


class ComprehensiveFineTuner:
    """
    Fine-tunes profitable traders by testing variations around optimal configs.
    Detects overfitting by comparing nearby parameter performance.
    """
    
    def __init__(
        self, 
        batch_results_dir: str = None,
        traders: List[str] = None,
        mode: str = 'standard'
    ):
        """
        Initialize fine-tuner.
        
        Args:
            batch_results_dir: Path to batch test results (auto-detect if None)
            traders: Specific traders to fine-tune (default: all profitable)
            mode: 'tight' (+-10%), 'standard' (+-20%), 'wide' (+-30%)
        """
        self.batch_results_dir = self._find_batch_results(batch_results_dir)
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/fine_tune_results/finetune_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load batch results
        self.batch_results = self._load_batch_results()
        
        # Filter to profitable traders or specified traders
        if traders:
            self.traders_to_tune = traders
        else:
            self.traders_to_tune = self._get_profitable_traders()
        
        self.fine_tune_results = {}
        
        logging.info("=" * 100)
        logging.info("COMPREHENSIVE FINE-TUNER INITIALIZED")
        logging.info("=" * 100)
        logging.info(f"Batch results from: {self.batch_results_dir}")
        logging.info(f"Mode: {mode.upper()} (+-{self._get_variation_pct()}%)")
        logging.info(f"Traders to fine-tune: {self.traders_to_tune}")
        logging.info(f"Output: {self.output_dir}")
        logging.info("=" * 100)
    
    def _find_batch_results(self, specified_dir: str = None) -> Path:
        """Find the most recent batch results directory."""
        if specified_dir:
            return Path(specified_dir)
        
        results_base = Path("backtester/optimization_results")
        if not results_base.exists():
            raise FileNotFoundError("No optimization results found. Run batch_test_all_traders.py first.")
        
        # Find most recent batch_* directory
        batch_dirs = sorted(results_base.glob("batch_*"), reverse=True)
        if not batch_dirs:
            raise FileNotFoundError("No batch results found. Run batch_test_all_traders.py first.")
        
        return batch_dirs[0]
    
    def _load_batch_results(self) -> Dict:
        """Load results from batch testing."""
        results = {}
        
        # Load master comparison
        master_csv = self.batch_results_dir / "master_comparison.csv"
        if master_csv.exists():
            results['master'] = pd.read_csv(master_csv)
        
        # Load individual trader results
        for trader_dir in self.batch_results_dir.iterdir():
            if trader_dir.is_dir():
                trader_name = trader_dir.name
                main_results = trader_dir / "main_all_results.csv"
                if main_results.exists():
                    df = pd.read_csv(main_results)
                    if len(df) > 0:
                        best_config = df.loc[df['total_pnl'].idxmax()].to_dict()
                        results[trader_name] = {
                            'all_results': df,
                            'best_config': best_config,
                            'best_pnl': best_config['total_pnl']
                        }
        
        return results
    
    def _get_profitable_traders(self) -> List[str]:
        """Get list of profitable traders from batch results."""
        profitable = []
        
        for trader, data in self.batch_results.items():
            if trader == 'master':
                continue
            if isinstance(data, dict) and data.get('best_pnl', 0) > 0:
                profitable.append(trader)
        
        return sorted(profitable, key=lambda t: self.batch_results[t]['best_pnl'], reverse=True)
    
    def _get_variation_pct(self) -> int:
        """Get variation percentage based on mode."""
        return {'tight': 10, 'standard': 20, 'wide': 30}[self.mode]
    
    def _create_fine_tune_grid(self, best_config: Dict) -> Dict:
        """
        Create fine-tuning grid around optimal configuration.
        Tests +-variation% around each optimal parameter.
        """
        variation = self._get_variation_pct() / 100
        
        def expand_numeric(value, variations=5, min_val=1, max_val=100, as_int=False):
            """Create range around a numeric value."""
            delta = max(value * variation, 1)
            values = np.linspace(value - delta, value + delta, variations)
            values = [max(min_val, min(max_val, v)) for v in values]
            if as_int:
                values = sorted(list(set([int(round(v)) for v in values])))
            else:
                values = sorted(list(set([round(v, 2) for v in values])))
            return values
        
        grid = {
            # Breakeven - critical parameter
            'breakeven_trigger_percent': expand_numeric(
                best_config['breakeven_trigger_percent'],
                variations=7, min_val=2, max_val=30, as_int=True
            ),
            
            # Trail method - test both if original is pullback
            'trail_method': ['pullback_percent', 'atr'] if best_config['trail_method'] == 'pullback_percent' else ['atr', 'pullback_percent'],
            
            # Pullback - fine-tune around optimal
            'pullback_percent': expand_numeric(
                best_config['pullback_percent'],
                variations=7, min_val=3, max_val=40, as_int=True
            ),
            
            # ATR period - test nearby values
            'atr_period': expand_numeric(
                best_config['atr_period'],
                variations=5, min_val=3, max_val=50, as_int=True
            ),
            
            # ATR multiplier - fine-tune
            'atr_multiplier': expand_numeric(
                best_config['atr_multiplier'],
                variations=7, min_val=0.3, max_val=5.0
            ),
            
            # Native trail - critical safety net
            'native_trail_percent': expand_numeric(
                best_config['native_trail_percent'],
                variations=7, min_val=5, max_val=50, as_int=True
            ),
            
            # PSAR - test on/off and nearby values
            'psar_enabled': [True, False],
            'psar_start': expand_numeric(
                best_config.get('psar_start', 0.02),
                variations=3, min_val=0.005, max_val=0.05
            ),
            'psar_increment': expand_numeric(
                best_config.get('psar_increment', 0.02),
                variations=3, min_val=0.005, max_val=0.05
            ),
            'psar_max': expand_numeric(
                best_config.get('psar_max', 0.2),
                variations=3, min_val=0.1, max_val=0.4
            ),
            
            # RSI - test on/off and nearby values
            'rsi_hook_enabled': [True, False],
            'rsi_period': expand_numeric(
                best_config.get('rsi_period', 14),
                variations=5, min_val=5, max_val=30, as_int=True
            ),
            'rsi_overbought': expand_numeric(
                best_config.get('rsi_overbought', 70),
                variations=5, min_val=60, max_val=85, as_int=True
            ),
            'rsi_oversold': expand_numeric(
                best_config.get('rsi_oversold', 30),
                variations=5, min_val=15, max_val=40, as_int=True
            ),
        }
        
        return grid
    
    def _count_combinations(self, grid: Dict) -> int:
        """Count total combinations in grid."""
        count = 1
        for values in grid.values():
            count *= len(values)
        return count
    
    def _generate_combinations(self, grid: Dict) -> List[Dict]:
        """Generate all parameter combinations."""
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    async def fine_tune_trader(self, trader_name: str) -> Dict:
        """
        Fine-tune a single trader.
        
        Args:
            trader_name: Name of trader to fine-tune
            
        Returns:
            Dictionary with fine-tuning results and stability analysis
        """
        logging.info("\n" + "=" * 100)
        logging.info(f"[TARGET] FINE-TUNING: {trader_name.upper()}")
        logging.info("=" * 100)
        
        # Get original best config
        if trader_name not in self.batch_results:
            logging.warning(f"No batch results found for {trader_name}")
            return self._empty_result(trader_name, "No batch results")
        
        trader_data = self.batch_results[trader_name]
        original_best = trader_data['best_config']
        original_pnl = trader_data['best_pnl']
        
        logging.info(f"[CHART] Original best P&L: ${original_pnl:.2f}")
        logging.info(f"[CHART] Original config: BE={original_best['breakeven_trigger_percent']}%, "
                    f"Trail={original_best['trail_method']}, Native={original_best['native_trail_percent']}%")
        
        # Find signals file
        signals_file = self._find_signals_file(trader_name)
        if not signals_file:
            return self._empty_result(trader_name, "No signals file")
        
        # Create fine-tune grid
        fine_tune_grid = self._create_fine_tune_grid(original_best)
        total_combinations = self._count_combinations(fine_tune_grid)
        
        logging.info(f"[WRENCH] Fine-tune grid: {total_combinations:,} combinations")
        
        # Create output directory
        trader_output = self.output_dir / trader_name
        trader_output.mkdir(exist_ok=True)
        
        # Run fine-tuning
        results = await self._run_fine_tune(signals_file, fine_tune_grid, trader_output)
        
        if not results:
            return self._empty_result(trader_name, "No results from fine-tuning")
        
        # Analyze results
        analysis = self._analyze_fine_tune_results(
            trader_name, results, original_best, original_pnl, trader_output
        )
        
        self.fine_tune_results[trader_name] = analysis
        
        return analysis
    
    def _find_signals_file(self, trader_name: str) -> Optional[Path]:
        """Find signals file for trader."""
        possible_paths = [
            Path(f"backtester/channel_signals/{trader_name}_signals.txt"),
            Path(f"backtester/channel_signals/{trader_name.lower()}_signals.txt"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    async def _run_fine_tune(
        self, 
        signals_file: Path, 
        grid: Dict,
        output_dir: Path
    ) -> List[Dict]:
        """Run fine-tuning tests."""
        combinations = self._generate_combinations(grid)
        total = len(combinations)
        results = []
        
        start_time = datetime.now()
        
        for i, params in enumerate(combinations, 1):
            if i % 100 == 0 or i == 1:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate / 60 if rate > 0 else 0
                logging.info(f"  [{i:,}/{total:,}] {rate:.1f}/sec - ETA: {eta:.1f} min")
            
            try:
                engine = BacktestEngine(
                    signal_file_path=str(signals_file),
                    data_folder_path="backtester/historical_data"
                )
                
                result = engine.run_simulation(params)
                
                if result and result.get('total_trades', 0) > 0:
                    results.append({
                        'total_trades': result['total_trades'],
                        'total_pnl': result['total_pnl'],
                        'win_rate': result['win_rate'],
                        'profit_factor': result.get('profit_factor', 0),
                        'avg_minutes_held': result.get('avg_minutes_held', 0),
                        **params
                    })
            except Exception as e:
                continue
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_dir / "fine_tune_all_results.csv", index=False, encoding='utf-8')
        
        return results
    
    def _analyze_fine_tune_results(
        self,
        trader_name: str,
        results: List[Dict],
        original_best: Dict,
        original_pnl: float,
        output_dir: Path
    ) -> Dict:
        """Analyze fine-tuning results for stability and improvement."""
        
        df = pd.DataFrame(results)
        
        # Find new best
        new_best = df.loc[df['total_pnl'].idxmax()].to_dict()
        new_best_pnl = new_best['total_pnl']
        
        # Calculate improvement
        improvement = new_best_pnl - original_pnl
        improvement_pct = (improvement / abs(original_pnl) * 100) if original_pnl != 0 else 0
        
        # Stability analysis - how many nearby configs are also profitable?
        profitable_configs = len(df[df['total_pnl'] > 0])
        total_configs = len(df)
        stability_score = profitable_configs / total_configs * 100
        
        # Calculate P&L variance around optimal
        top_10_pct = df.nlargest(int(len(df) * 0.1), 'total_pnl')
        pnl_variance = top_10_pct['total_pnl'].std()
        
        # Detect overfitting - if optimal is much better than neighbors
        optimal_pnl = new_best_pnl
        avg_top_10_pnl = top_10_pct['total_pnl'].mean()
        overfit_score = (optimal_pnl - avg_top_10_pnl) / abs(avg_top_10_pnl) * 100 if avg_top_10_pnl != 0 else 0
        
        # Parameter sensitivity in fine-tune range
        param_stability = self._analyze_param_stability(df)
        
        # Generate stability verdict
        if stability_score > 70 and overfit_score < 20:
            verdict = "[OK] ROBUST - Config is stable across parameter variations"
        elif stability_score > 50 and overfit_score < 30:
            verdict = "[+] ACCEPTABLE - Config is reasonably stable"
        elif stability_score > 30:
            verdict = "[!] SENSITIVE - Config is parameter-sensitive, use caution"
        else:
            verdict = "[X] UNSTABLE - Config may be overfit, original results unreliable"
        
        # Generate detailed report
        self._generate_fine_tune_report(
            trader_name, df, original_best, original_pnl, new_best, new_best_pnl,
            improvement, improvement_pct, stability_score, overfit_score,
            param_stability, verdict, output_dir
        )
        
        return {
            'trader': trader_name,
            'original_pnl': original_pnl,
            'new_best_pnl': new_best_pnl,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'stability_score': stability_score,
            'overfit_score': overfit_score,
            'profitable_configs': profitable_configs,
            'total_configs': total_configs,
            'new_best_config': new_best,
            'param_stability': param_stability,
            'verdict': verdict
        }
    
    def _analyze_param_stability(self, df: pd.DataFrame) -> Dict:
        """Analyze how stable P&L is across parameter changes."""
        stability = {}
        
        params = [
            'breakeven_trigger_percent', 'pullback_percent', 
            'atr_multiplier', 'native_trail_percent'
        ]
        
        for param in params:
            if param in df.columns:
                grouped = df.groupby(param)['total_pnl'].agg(['mean', 'std'])
                pnl_range = grouped['mean'].max() - grouped['mean'].min()
                avg_std = grouped['std'].mean()
                
                # Lower range = more stable
                if pnl_range < 50:
                    rating = "STABLE"
                elif pnl_range < 100:
                    rating = "MODERATE"
                else:
                    rating = "SENSITIVE"
                
                stability[param] = {
                    'pnl_range': pnl_range,
                    'avg_std': avg_std,
                    'rating': rating
                }
        
        return stability
    
    def _generate_fine_tune_report(
        self, trader_name: str, df: pd.DataFrame,
        original_best: Dict, original_pnl: float,
        new_best: Dict, new_best_pnl: float,
        improvement: float, improvement_pct: float,
        stability_score: float, overfit_score: float,
        param_stability: Dict, verdict: str,
        output_dir: Path
    ):
        """Generate detailed fine-tuning report."""
        
        report = []
        report.append("=" * 100)
        report.append(f"FINE-TUNING REPORT: {trader_name.upper()}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 100)
        
        # Comparison
        report.append("\n" + "-" * 50)
        report.append("ORIGINAL vs FINE-TUNED")
        report.append("-" * 50)
        report.append(f"Original Best P&L: ${original_pnl:.2f}")
        report.append(f"Fine-Tuned Best P&L: ${new_best_pnl:.2f}")
        report.append(f"Improvement: ${improvement:.2f} ({improvement_pct:+.1f}%)")
        
        # Stability metrics
        report.append("\n" + "-" * 50)
        report.append("STABILITY ANALYSIS")
        report.append("-" * 50)
        report.append(f"Stability Score: {stability_score:.1f}% (% of configs profitable)")
        report.append(f"Overfit Score: {overfit_score:.1f}% (lower is better, <20% is good)")
        report.append(f"Total configs tested: {len(df):,}")
        report.append(f"Profitable configs: {len(df[df['total_pnl'] > 0]):,}")
        
        # Verdict
        report.append("\n" + "-" * 50)
        report.append("VERDICT")
        report.append("-" * 50)
        report.append(verdict)
        
        # Parameter stability
        report.append("\n" + "-" * 50)
        report.append("PARAMETER STABILITY")
        report.append("-" * 50)
        for param, data in param_stability.items():
            report.append(f"{param}:")
            report.append(f"  P&L Range: ${data['pnl_range']:.2f}")
            report.append(f"  Rating: {data['rating']}")
        
        # New best config
        report.append("\n" + "-" * 50)
        report.append("FINE-TUNED OPTIMAL CONFIG")
        report.append("-" * 50)
        report.append(f"P&L: ${new_best_pnl:.2f}")
        report.append(f"Win Rate: {new_best['win_rate']:.1f}%")
        report.append(f"Profit Factor: {new_best['profit_factor']:.2f}")
        report.append(f"\nParameters:")
        report.append(f"  breakeven_trigger_percent: {new_best['breakeven_trigger_percent']}")
        report.append(f"  trail_method: {new_best['trail_method']}")
        report.append(f"  pullback_percent: {new_best['pullback_percent']}")
        report.append(f"  atr_period: {new_best['atr_period']}")
        report.append(f"  atr_multiplier: {new_best['atr_multiplier']}")
        report.append(f"  native_trail_percent: {new_best['native_trail_percent']}")
        report.append(f"  psar_enabled: {new_best['psar_enabled']}")
        report.append(f"  rsi_hook_enabled: {new_best['rsi_hook_enabled']}")
        
        # Top 10 fine-tuned configs
        report.append("\n" + "-" * 50)
        report.append("TOP 10 FINE-TUNED CONFIGS")
        report.append("-" * 50)
        top_10 = df.nlargest(10, 'total_pnl')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            report.append(f"{i:2d}. ${row['total_pnl']:8.2f} | WR: {row['win_rate']:5.1f}% | "
                         f"BE: {row['breakeven_trigger_percent']:2.0f}% | "
                         f"Pull: {row['pullback_percent']:2.0f}% | "
                         f"Native: {row['native_trail_percent']:2.0f}%")
        
        # Save report
        report_path = output_dir / f"{trader_name}_fine_tune_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logging.info(f"  [DOC] Report: {report_path}")
    
    def _empty_result(self, trader_name: str, reason: str) -> Dict:
        """Return empty result structure."""
        return {
            'trader': trader_name,
            'original_pnl': 0,
            'new_best_pnl': 0,
            'improvement': 0,
            'improvement_pct': 0,
            'stability_score': 0,
            'overfit_score': 100,
            'profitable_configs': 0,
            'total_configs': 0,
            'new_best_config': {},
            'param_stability': {},
            'verdict': f"[X] SKIP - {reason}"
        }
    
    def generate_master_report(self):
        """Generate master fine-tuning comparison report."""
        
        if not self.fine_tune_results:
            logging.warning("No fine-tune results to report")
            return
        
        report = []
        report.append("=" * 100)
        report.append("MASTER FINE-TUNING REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Mode: {self.mode.upper()}")
        report.append("=" * 100)
        
        # Sort by new best P&L
        sorted_results = sorted(
            self.fine_tune_results.values(),
            key=lambda x: x['new_best_pnl'],
            reverse=True
        )
        
        report.append("\n" + "-" * 100)
        report.append("TRADER RANKINGS (Post Fine-Tuning)")
        report.append("-" * 100)
        
        for r in sorted_results:
            stability_icon = "[OK]" if r['stability_score'] > 50 else "[!]" if r['stability_score'] > 30 else "[X]"
            report.append(
                f"{stability_icon} {r['trader'].upper():15s} | "
                f"New P&L: ${r['new_best_pnl']:8.2f} | "
                f"Original: ${r['original_pnl']:8.2f} | "
                f"Delta: ${r['improvement']:+7.2f} | "
                f"Stability: {r['stability_score']:5.1f}%"
            )
        
        # Recommendations
        report.append("\n" + "-" * 100)
        report.append("RECOMMENDATIONS")
        report.append("-" * 100)
        
        robust = [r for r in sorted_results if r['stability_score'] > 50 and r['new_best_pnl'] > 0]
        if robust:
            report.append("\n[OK] ROBUST (Ready for Production):")
            for r in robust:
                report.append(f"   - {r['trader'].upper()}: ${r['new_best_pnl']:.2f} ({r['stability_score']:.0f}% stable)")
        
        marginal = [r for r in sorted_results if 30 < r['stability_score'] <= 50 and r['new_best_pnl'] > 0]
        if marginal:
            report.append("\n[!] MARGINAL (Paper Trade First):")
            for r in marginal:
                report.append(f"   - {r['trader'].upper()}: ${r['new_best_pnl']:.2f} ({r['stability_score']:.0f}% stable)")
        
        avoid = [r for r in sorted_results if r['stability_score'] <= 30 or r['new_best_pnl'] <= 0]
        if avoid:
            report.append("\n[X] AVOID (Unstable/Unprofitable):")
            for r in avoid:
                report.append(f"   - {r['trader'].upper()}: ${r['new_best_pnl']:.2f} ({r['stability_score']:.0f}% stable)")
        
        # Save report
        report_path = self.output_dir / "MASTER_FINE_TUNE_REPORT.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Save CSV
        df = pd.DataFrame(sorted_results)
        df.to_csv(self.output_dir / "master_fine_tune_comparison.csv", index=False, encoding='utf-8')
        
        logging.info(f"\n[CHART] Master report: {report_path}")
    
    async def run(self):
        """Run fine-tuning on all profitable traders."""
        
        logging.info("\n" + "=" * 100)
        logging.info("STARTING COMPREHENSIVE FINE-TUNING")
        logging.info("=" * 100 + "\n")
        
        start_time = datetime.now()
        
        for i, trader in enumerate(self.traders_to_tune, 1):
            logging.info(f"\n[{i}/{len(self.traders_to_tune)}] Fine-tuning {trader}...")
            await self.fine_tune_trader(trader)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate master report
        self.generate_master_report()
        
        logging.info("\n" + "=" * 100)
        logging.info("FINE-TUNING COMPLETE!")
        logging.info(f"Total time: {duration}")
        logging.info(f"Results: {self.output_dir}")
        logging.info("=" * 100)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Comprehensive Fine-Tuner')
    parser.add_argument('--batch-dir', type=str, default=None,
                       help='Path to batch test results (auto-detect if not specified)')
    parser.add_argument('--traders', nargs='+', default=None,
                       help='Specific traders to fine-tune (default: all profitable)')
    parser.add_argument('--mode', choices=['tight', 'standard', 'wide'], default='standard',
                       help='Fine-tune variation: tight (+-10%%), standard (+-20%%), wide (+-30%%)')
    
    args = parser.parse_args()
    
    tuner = ComprehensiveFineTuner(
        batch_results_dir=args.batch_dir,
        traders=args.traders,
        mode=args.mode
    )
    await tuner.run()


if __name__ == "__main__":
    asyncio.run(main())
