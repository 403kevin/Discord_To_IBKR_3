#!/usr/bin/env python3
"""
batch_fine_tune.py - Fine-Tune Optimizer for Profitable Traders
================================================================
Reads best configurations from initial batch test and creates tight
parameter grids around winners to find absolute optimal settings.

Usage:
    python backtester/batch_fine_tune.py --source BATCH_TEST_20251023_060602
    python backtester/batch_fine_tune.py --source BATCH_TEST_20251023_060602 --traders qiqo zeus
    python backtester/batch_fine_tune.py --source BATCH_TEST_20251023_060602 --steps 1  # Tighter grid
    
Time: ~5-10 minutes per trader (27-125 tests vs 3,607)
"""

import asyncio
import logging
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse

# Add project root
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'backtester' else Path(__file__).parent
sys.path.insert(0, str(project_root))

from backtester.backtest_engine import BacktestEngine
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class BatchFineTuner:
    """
    Fine-tune optimizer - creates tight grids around best configs
    """
    
    def __init__(self, source_dir: str, steps: int = 2, trader_list: List[str] = None):
        self.source_dir = Path(f"backtester/optimization_results/{source_dir}")
        self.steps = steps  # How many steps ±
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"backtester/optimization_results/FINE_TUNE_{self.timestamp}")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load best configs from source
        self.best_configs = self._load_best_configs(trader_list)
        
        self.master_results = []
        
        logging.info("="*80)
        logging.info("🔬 BATCH FINE-TUNE OPTIMIZER")
        logging.info("="*80)
        logging.info(f"Source: {self.source_dir}")
        logging.info(f"Grid steps: ±{steps}")
        logging.info(f"Traders to fine-tune: {len(self.best_configs)}")
        logging.info(f"Output: {self.output_dir}")
        logging.info("="*80)
    
    def _load_best_configs(self, trader_filter: List[str] = None) -> Dict:
        """Load best configurations from source batch test"""
        
        master_csv = self.source_dir / "MASTER_COMPARISON.csv"
        
        if not master_csv.exists():
            logging.error(f"Source not found: {master_csv}")
            sys.exit(1)
        
        df = pd.read_csv(master_csv)
        
        # Filter to only profitable traders
        df = df[df['best_pnl'] > 0]
        
        # Apply trader filter if specified
        if trader_filter:
            df = df[df['trader'].isin(trader_filter)]
        
        if df.empty:
            logging.error("No profitable traders found in source")
            sys.exit(1)
        
        best_configs = {}
        
        for _, row in df.iterrows():
            trader = row['trader']
            
            # Load full results to get best config details
            trader_csv = self.source_dir / trader / f"{trader}_all_results.csv"
            
            if not trader_csv.exists():
                logging.warning(f"Results not found for {trader}: {trader_csv}")
                continue
            
            trader_df = pd.read_csv(trader_csv)
            
            # Get best config by P&L
            best_row = trader_df.loc[trader_df['total_pnl'].idxmax()]
            
            best_configs[trader] = {
                'breakeven_trigger_percent': int(best_row['breakeven_trigger_percent']),
                'trail_method': best_row['trail_method'],
                'pullback_percent': int(best_row['pullback_percent']),
                'atr_period': int(best_row['atr_period']),
                'atr_multiplier': float(best_row['atr_multiplier']),
                'native_trail_percent': int(best_row['native_trail_percent']),
                'best_pnl': best_row['total_pnl'],
                'best_win_rate': best_row['win_rate']
            }
            
            logging.info(f"✅ Loaded best config for {trader}: ${best_row['total_pnl']:.2f}")
        
        return best_configs
    
    def _create_fine_tune_grid(self, best_config: Dict) -> Dict:
        """
        Create tight parameter grid around best config
        Grid size depends on self.steps (1 = tight, 2 = medium, 3 = wide)
        """
        
        # Helper to create range around value
        def range_around(value, options, steps):
            """Get ±steps around value from options list"""
            if value not in options:
                # Find closest
                closest = min(options, key=lambda x: abs(x - value))
                idx = options.index(closest)
            else:
                idx = options.index(value)
            
            start = max(0, idx - steps)
            end = min(len(options), idx + steps + 1)
            return options[start:end]
        
        # Define full option spaces
        breakeven_options = [5, 7, 10, 12, 15]
        pullback_options = [8, 10, 12, 15]
        atr_period_options = [5, 10, 14, 20, 30]
        atr_multiplier_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        native_options = [25, 30, 35]
        
        # Create tight grid
        grid = {
            'breakeven_trigger_percent': range_around(
                best_config['breakeven_trigger_percent'], 
                breakeven_options, 
                self.steps
            ),
            'trail_method': [best_config['trail_method']],  # Keep best method
            'pullback_percent': range_around(
                best_config['pullback_percent'],
                pullback_options,
                self.steps
            ),
            'atr_period': range_around(
                best_config['atr_period'],
                atr_period_options,
                self.steps
            ),
            'atr_multiplier': range_around(
                best_config['atr_multiplier'],
                atr_multiplier_options,
                self.steps
            ),
            'native_trail_percent': range_around(
                best_config['native_trail_percent'],
                native_options,
                self.steps
            )
        }
        
        # Calculate test count
        test_count = 1
        for values in grid.values():
            test_count *= len(values)
        
        return grid, test_count
    
    async def fine_tune_trader(self, trader: str, best_config: Dict) -> Dict:
        """Fine-tune a single trader"""
        
        logging.info("\n" + "="*80)
        logging.info(f"🔬 FINE-TUNING: {trader.upper()}")
        logging.info("="*80)
        logging.info(f"Original best P&L: ${best_config['best_pnl']:.2f}")
        logging.info(f"Original best WR: {best_config['best_win_rate']:.1f}%")
        
        signals_file = Path(f"backtester/channel_signals/{trader}_signals.txt")
        
        if not signals_file.exists():
            logging.warning(f"⚠️ Signals file not found: {signals_file}")
            return self._empty_result(trader, best_config)
        
        trader_output_dir = self.output_dir / trader
        trader_output_dir.mkdir(exist_ok=True)
        
        # Create fine-tune grid
        param_grid, test_count = self._create_fine_tune_grid(best_config)
        
        logging.info(f"\n📊 Fine-tune grid:")
        for param, values in param_grid.items():
            logging.info(f"  {param}: {values}")
        logging.info(f"\nTotal tests: {test_count}")
        logging.info(f"Est time: ~{test_count * 2 / 60:.0f} minutes\n")
        
        # Generate combinations
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combinations = list(product(*values))
        
        results = []
        
        # Test each combination
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            
            if idx % 10 == 0 or idx == 1:
                logging.info(f"  Progress: {idx}/{test_count} ({idx/test_count*100:.1f}%)")
            
            try:
                engine = BacktestEngine(str(signals_file))
                result = engine.run_simulation(params)
                
                result['trader'] = trader
                result.update(params)
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error testing {trader} combo {idx}: {e}")
                continue
        
        if not results:
            logging.warning(f"⚠️ No results for {trader}")
            return self._empty_result(trader, best_config)
        
        # Save results
        df = pd.DataFrame(results)
        results_csv = trader_output_dir / f"{trader}_fine_tune_results.csv"
        df.to_csv(results_csv, index=False)
        
        # Find new best
        new_best = df.loc[df['total_pnl'].idxmax()]
        
        improvement = new_best['total_pnl'] - best_config['best_pnl']
        improvement_pct = (improvement / abs(best_config['best_pnl'])) * 100 if best_config['best_pnl'] != 0 else 0
        
        logging.info(f"\n🎯 FINE-TUNE RESULTS:")
        logging.info(f"  Original P&L: ${best_config['best_pnl']:.2f}")
        logging.info(f"  New best P&L: ${new_best['total_pnl']:.2f}")
        logging.info(f"  Improvement: ${improvement:.2f} ({improvement_pct:+.1f}%)")
        logging.info(f"  New win rate: {new_best['win_rate']:.1f}%")
        
        # Generate summary
        summary = self._generate_fine_tune_summary(
            trader, 
            best_config, 
            new_best, 
            df, 
            trader_output_dir
        )
        
        # Add to master results
        self.master_results.append({
            'trader': trader,
            'original_pnl': best_config['best_pnl'],
            'new_pnl': new_best['total_pnl'],
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'new_win_rate': new_best['win_rate'],
            'tests_run': test_count
        })
        
        return summary
    
    def _generate_fine_tune_summary(self, trader: str, original: Dict, 
                                     new_best: pd.Series, df: pd.DataFrame, 
                                     output_dir: Path) -> Dict:
        """Generate fine-tune summary report"""
        
        improvement = new_best['total_pnl'] - original['best_pnl']
        improvement_pct = (improvement / abs(original['best_pnl'])) * 100 if original['best_pnl'] != 0 else 0
        
        summary_text = f"""
{'='*80}
FINE-TUNE RESULTS: {trader.upper()}
{'='*80}

📊 COMPARISON:
{'─'*80}
Original Best P&L:       ${original['best_pnl']:,.2f}
New Best P&L:            ${new_best['total_pnl']:,.2f}
Improvement:             ${improvement:,.2f} ({improvement_pct:+.1f}%)

Original Win Rate:       {original['best_win_rate']:.1f}%
New Win Rate:            {new_best['win_rate']:.1f}%

{'─'*80}
🏆 NEW OPTIMAL CONFIGURATION:
{'─'*80}
Breakeven trigger:       {new_best['breakeven_trigger_percent']}%
Trail method:            {new_best['trail_method']}
Pullback percent:        {new_best['pullback_percent']}%
ATR period:              {new_best['atr_period']}
ATR multiplier:          {new_best['atr_multiplier']}
Native trail:            {new_best['native_trail_percent']}%

Total trades:            {new_best['total_trades']:.0f}
Profit factor:           {new_best['profit_factor']:.2f}

{'─'*80}
📈 TOP 5 FINE-TUNED CONFIGURATIONS:
{'─'*80}
"""
        
        top5 = df.nlargest(5, 'total_pnl')
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            summary_text += f"\n#{idx}. ${row['total_pnl']:,.2f} | WR: {row['win_rate']:.1f}% | "
            summary_text += f"BE: {row['breakeven_trigger_percent']}% | "
            summary_text += f"PB: {row['pullback_percent']}% | "
            summary_text += f"Native: {row['native_trail_percent']}%"
        
        summary_text += f"\n\n{'='*80}\n"
        
        # Save summary
        summary_file = output_dir / f"{trader}_fine_tune_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        return {
            'trader': trader,
            'original_pnl': original['best_pnl'],
            'new_pnl': new_best['total_pnl'],
            'improvement': improvement
        }
    
    def _empty_result(self, trader: str, best_config: Dict) -> Dict:
        """Return empty result if fine-tune fails"""
        return {
            'trader': trader,
            'original_pnl': best_config['best_pnl'],
            'new_pnl': 0,
            'improvement': 0
        }
    
    async def run_fine_tuning(self):
        """Run fine-tuning on all profitable traders"""
        
        logging.info("\n" + "🔬"*40)
        logging.info("STARTING BATCH FINE-TUNING")
        logging.info("🔬"*40 + "\n")
        
        start_time = datetime.now()
        
        for trader, best_config in self.best_configs.items():
            await self.fine_tune_trader(trader, best_config)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate master report
        self._generate_master_fine_tune_report(duration)
        
        logging.info("\n" + "🎉"*40)
        logging.info("FINE-TUNING COMPLETE!")
        logging.info("🎉"*40 + "\n")
    
    def _generate_master_fine_tune_report(self, duration):
        """Generate master fine-tune comparison report"""
        
        if not self.master_results:
            logging.warning("No fine-tune results to report")
            return
        
        df = pd.DataFrame(self.master_results)
        df = df.sort_values('improvement', ascending=False)
        
        # Save CSV
        master_csv = self.output_dir / "FINE_TUNE_COMPARISON.csv"
        df.to_csv(master_csv, index=False)
        
        # Generate report
        report = f"""
{'='*80}
🔬 FINE-TUNE MASTER REPORT
{'='*80}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}
Traders Fine-Tuned: {len(df)}

{'='*80}
📊 IMPROVEMENT RANKINGS:
{'='*80}

"""
        
        for idx, row in df.iterrows():
            rank = df.index.get_loc(idx) + 1
            
            report += f"""
{'─'*80}
#{rank}. {row['trader'].upper()}
{'─'*80}
Original P&L:        ${row['original_pnl']:,.2f}
Fine-tuned P&L:      ${row['new_pnl']:,.2f}
Improvement:         ${row['improvement']:,.2f} ({row['improvement_pct']:+.1f}%)
New win rate:        {row['new_win_rate']:.1f}%
Tests run:           {row['tests_run']:.0f}

"""
        
        total_improvement = df['improvement'].sum()
        avg_improvement = df['improvement'].mean()
        
        report += f"""
{'='*80}
📈 SUMMARY:
{'='*80}

Total improvement:       ${total_improvement:,.2f}
Average improvement:     ${avg_improvement:,.2f}
Traders improved:        {len(df[df['improvement'] > 0])}
Traders declined:        {len(df[df['improvement'] < 0])}

{'='*80}
📁 OUTPUT FILES:
{'='*80}

Master comparison:       {master_csv}
Individual reports:      {self.output_dir}/<trader>/<trader>_fine_tune_summary.txt
Full results CSVs:       {self.output_dir}/<trader>/<trader>_fine_tune_results.csv

{'='*80}
"""
        
        # Save report
        report_file = self.output_dir / "FINE_TUNE_MASTER_REPORT.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print to console
        print(report)
        
        logging.info(f"✅ Fine-tune master report saved: {report_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune optimizer for profitable traders"
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Source batch test directory name (e.g., BATCH_TEST_20251023_060602)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=2,
        help='Grid steps (1=tight, 2=medium, 3=wide). Default: 2'
    )
    parser.add_argument(
        '--traders',
        nargs='+',
        help='Specific traders to fine-tune (default: all profitable)'
    )
    
    args = parser.parse_args()
    
    # Create fine-tuner
    fine_tuner = BatchFineTuner(
        source_dir=args.source,
        steps=args.steps,
        trader_list=args.traders
    )
    
    # Run fine-tuning
    await fine_tuner.run_fine_tuning()


if __name__ == "__main__":
    asyncio.run(main())
