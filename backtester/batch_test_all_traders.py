#!/usr/bin/env python3
"""
batch_test_all_traders.py - COMPREHENSIVE BATCH TESTING SYSTEM
==============================================================
Tests ALL traders with full parameter optimization including:
- Full dynamic exit tests (3,607 combinations)
- Native-trail-only curiosity tests (7 combinations)  
- Master comparison reports
- Trader scorecard rankings
- All results logged to CSV

Usage:
    python batch_test_all_traders.py                    # Full mode (all 3,614 tests per trader)
    python batch_test_all_traders.py --quick             # Quick mode (16 tests per trader)
    python batch_test_all_traders.py --traders goldman expo qiqo  # Test specific traders only
    
Time Estimates:
    Quick mode: ~10 min per trader × 11 traders = ~2 hours total
    Full mode: ~2 hours per trader × 11 traders = ~22 hours total
"""

import asyncio
import logging
import sys
import json
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


class BatchTesterAllTraders:
    """
    Comprehensive batch testing system for all Discord traders.
    Tests each trader with full parameter optimization.
    """
    
    def __init__(self, quick_mode=False, trader_list=None):
        self.quick_mode = quick_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_output_dir = Path(f"backtester/optimization_results/BATCH_TEST_{self.timestamp}")
        self.master_output_dir.mkdir(exist_ok=True, parents=True)
        
        # All traders to test
        self.all_traders = trader_list or [
            'goldman', 'expo', 'qiqo', 'gandalf', 'zeus', 
            'arrow', 'waxui', 'nitro', 'money_mo', 'diesel', 'prophet', 'dd_alerts'
        ]
        
        self.master_results = []
        self.trader_scorecards = {}
        
        logging.info("="*80)
        logging.info("🚀 COMPREHENSIVE BATCH TESTING SYSTEM")
        logging.info("="*80)
        logging.info(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
        logging.info(f"Traders: {', '.join(self.all_traders)}")
        logging.info(f"Output: {self.master_output_dir}")
        logging.info("="*80)
    
    def get_quick_grid(self) -> Dict:
        """Quick test parameters - 16 combinations"""
        return {
            'breakeven_trigger_percent': [7, 10, 12],
            'trail_method': ['pullback_percent', 'atr'],
            'pullback_percent': [10, 12],
            'atr_period': [14],
            'atr_multiplier': [1.5, 2.0],
            'native_trail_percent': [25, 30]
        }
    
    def get_full_grid(self) -> Dict:
        """Full test parameters - 3,607 combinations total"""
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
            'native_trail_percent': [10, 12, 15, 17, 20, 25, 30]  # 7 tests
        }
    
    async def test_single_trader(self, trader_name: str) -> Dict:
        """
        Test a single trader with full optimization
        Returns summary statistics for master comparison
        """
        logging.info("\n" + "="*80)
        logging.info(f"🎯 TESTING: {trader_name.upper()}")
        logging.info("="*80)
        
        signals_file = Path(f"backtester/channel_signals/{trader_name}_signals.txt")
        
        if not signals_file.exists():
            logging.warning(f"⚠️ Signals file not found: {signals_file}")
            return self._empty_trader_result(trader_name)
        
        trader_output_dir = self.master_output_dir / trader_name
        trader_output_dir.mkdir(exist_ok=True)
        
        results = []
        
        # 1. Test dynamic exits (full grid)
        if self.quick_mode:
            param_grid = self.get_quick_grid()
        else:
            param_grid = self.get_full_grid()
        
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combinations = list(product(*values))
        
        total_dynamic_tests = len(combinations)
        
        logging.info(f"\n📊 DYNAMIC EXIT TESTS: {total_dynamic_tests} combinations")
        
        for idx, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            
            if idx % 100 == 0 or idx == 1:
                logging.info(f"  Progress: {idx}/{total_dynamic_tests} ({idx/total_dynamic_tests*100:.1f}%)")
            
            try:
                engine = BacktestEngine(str(signals_file))
                result = engine.run_simulation(params)
                
                result['test_type'] = 'dynamic_exit'
                result['trader'] = trader_name
                result.update(params)
                results.append(result)
                
            except Exception as e:
                logging.error(f"Error testing {trader_name} combo {idx}: {e}")
                continue
        
        # 2. Test native-trail-only (curiosity tests) - ONLY in full mode
        if not self.quick_mode:
            nt_grid = self.get_native_trail_only_grid()
            nt_keys = list(nt_grid.keys())
            nt_values = [nt_grid[k] for k in nt_keys]
            nt_combinations = list(product(*nt_values))
            
            total_nt_tests = len(nt_combinations)
            
            logging.info(f"\n📊 NATIVE-TRAIL-ONLY TESTS: {total_nt_tests} combinations")
            
            for idx, combo in enumerate(nt_combinations, 1):
                params = dict(zip(nt_keys, combo))
                
                logging.info(f"  Native trail test: {idx}/{total_nt_tests} (trail={params['native_trail_percent']}%)")
                
                try:
                    engine = BacktestEngine(str(signals_file))
                    result = engine.run_simulation(params)
                    
                    result['test_type'] = 'native_trail_only'
                    result['trader'] = trader_name
                    result.update(params)
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Error testing {trader_name} native trail {idx}: {e}")
                    continue
        
        # 3. Save trader results
        if results:
            df = pd.DataFrame(results)
            
            # Save full results CSV
            results_csv = trader_output_dir / f"{trader_name}_all_results.csv"
            df.to_csv(results_csv, index=False)
            logging.info(f"✅ Saved {len(results)} test results to {results_csv}")
            
            # Generate trader summary
            summary = self._generate_trader_summary(df, trader_name, trader_output_dir)
            
            # Add to master results
            self.master_results.append({
                'trader': trader_name,
                'total_tests': len(results),
                'best_pnl': summary['best_pnl'],
                'best_win_rate': summary['best_win_rate'],
                'best_profit_factor': summary['best_profit_factor'],
                'profitable_configs': summary['profitable_configs'],
                'best_config': summary['best_config'],
                'avg_trades': summary['avg_trades'],
                'recommendation': summary['recommendation']
            })
            
            return summary
        else:
            logging.warning(f"⚠️ No valid results for {trader_name}")
            return self._empty_trader_result(trader_name)
    
    def _generate_trader_summary(self, df: pd.DataFrame, trader_name: str, output_dir: Path) -> Dict:
        """Generate comprehensive summary for a single trader"""
        
        # Filter out zero-trade results
        df = df[df['total_trades'] > 0].copy()
        
        if df.empty:
            return self._empty_trader_result(trader_name)
        
        # Find best configurations
        best_pnl_row = df.loc[df['total_pnl'].idxmax()]
        best_wr_row = df.loc[df['win_rate'].idxmax()]
        
        # Count profitable configs
        profitable_configs = len(df[df['total_pnl'] > 0])
        
        # Calculate averages
        avg_pnl = df['total_pnl'].mean()
        avg_win_rate = df['win_rate'].mean()
        avg_trades = df['total_trades'].mean()
        
        # Separate dynamic vs native-trail-only results
        dynamic_results = df[df['test_type'] == 'dynamic_exit']
        native_only_results = df[df['test_type'] == 'native_trail_only']
        
        # Generate recommendation
        if best_pnl_row['total_pnl'] > 0 and best_pnl_row['win_rate'] > 50:
            recommendation = "✅ PROFITABLE - Recommend trading"
        elif profitable_configs > 0:
            recommendation = "⚠️ CONDITIONALLY PROFITABLE - Further testing needed"
        else:
            recommendation = "❌ UNPROFITABLE - Do not trade"
        
        # Build summary report
        summary_text = f"""
{'='*80}
TRADER: {trader_name.upper()}
{'='*80}

📊 OVERALL STATISTICS:
{'─'*80}
Total tests run:          {len(df)}
  - Dynamic exit tests:   {len(dynamic_results)}
  - Native-trail tests:   {len(native_only_results)}

Profitable configs:       {profitable_configs} ({profitable_configs/len(df)*100:.1f}%)
Average P&L:              ${avg_pnl:,.2f}
Average win rate:         {avg_win_rate:.1f}%
Average trades:           {avg_trades:.0f}

{'─'*80}
🏆 BEST CONFIGURATION (by P&L):
{'─'*80}
Total P&L:                ${best_pnl_row['total_pnl']:,.2f}
Win rate:                 {best_pnl_row['win_rate']:.1f}%
Profit factor:            {best_pnl_row['profit_factor']:.2f}
Total trades:             {best_pnl_row['total_trades']:.0f}
Max drawdown:             ${best_pnl_row['max_drawdown']:,.2f}

Exit Parameters:
  Breakeven trigger:      {best_pnl_row['breakeven_trigger_percent']}%
  Trail method:           {best_pnl_row['trail_method']}
  Pullback percent:       {best_pnl_row['pullback_percent']}%
  ATR period:             {best_pnl_row['atr_period']}
  ATR multiplier:         {best_pnl_row['atr_multiplier']}
  Native trail:           {best_pnl_row['native_trail_percent']}%

Exit Reason Breakdown:
  Breakeven stops:        {best_pnl_row.get('breakeven_exits', 0):.0f}
  Pullback stops:         {best_pnl_row.get('pullback_exits', 0):.0f}
  ATR stops:              {best_pnl_row.get('atr_exits', 0):.0f}
  Native trail:           {best_pnl_row.get('native_trail_exits', 0):.0f}
  EOD closes:             {best_pnl_row.get('eod_exits', 0):.0f}

{'─'*80}
📈 TOP 10 CONFIGURATIONS:
{'─'*80}
"""
        
        # Add top 10
        top10 = df.nlargest(10, 'total_pnl')
        for idx, row in top10.iterrows():
            summary_text += f"\n#{idx+1}. ${row['total_pnl']:,.2f} | WR: {row['win_rate']:.1f}% | "
            summary_text += f"BE: {row['breakeven_trigger_percent']}% | "
            summary_text += f"Trail: {row['trail_method']} | "
            summary_text += f"PB: {row['pullback_percent']}% | "
            summary_text += f"Native: {row['native_trail_percent']}%"
        
        # Add native-trail-only analysis if available
        if not native_only_results.empty:
            best_native = native_only_results.loc[native_only_results['total_pnl'].idxmax()]
            
            summary_text += f"""

{'─'*80}
🔍 NATIVE-TRAIL-ONLY ANALYSIS (Curiosity Tests):
{'─'*80}
These tests disable breakeven and pullback exits to see how native trail alone performs.

Best native-trail-only config:
  Native trail:           {best_native['native_trail_percent']}%
  Total P&L:              ${best_native['total_pnl']:,.2f}
  Win rate:               {best_native['win_rate']:.1f}%
  Profit factor:          {best_native['profit_factor']:.2f}
  Total trades:           {best_native['total_trades']:.0f}

All native-trail-only results:
"""
            for _, row in native_only_results.iterrows():
                summary_text += f"  {row['native_trail_percent']}% trail: ${row['total_pnl']:,.2f} | "
                summary_text += f"WR: {row['win_rate']:.1f}% | PF: {row['profit_factor']:.2f}\n"
        
        summary_text += f"""
{'─'*80}
💡 RECOMMENDATION:
{'─'*80}
{recommendation}

{'='*80}
"""
        
        # Save summary report
        summary_file = output_dir / f"{trader_name}_optimization_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        logging.info(f"✅ Generated summary report: {summary_file}")
        
        return {
            'trader': trader_name,
            'best_pnl': best_pnl_row['total_pnl'],
            'best_win_rate': best_wr_row['win_rate'],
            'best_profit_factor': best_pnl_row['profit_factor'],
            'profitable_configs': profitable_configs,
            'best_config': best_pnl_row.to_dict(),
            'avg_trades': avg_trades,
            'recommendation': recommendation
        }
    
    def _empty_trader_result(self, trader_name: str) -> Dict:
        """Return empty result structure for traders with no data"""
        return {
            'trader': trader_name,
            'best_pnl': 0,
            'best_win_rate': 0,
            'best_profit_factor': 0,
            'profitable_configs': 0,
            'best_config': {},
            'avg_trades': 0,
            'recommendation': "❌ NO DATA - Check signals file"
        }
    
    async def run_batch_tests(self):
        """Run batch tests on all traders"""
        
        logging.info("\n" + "🚀"*40)
        logging.info("STARTING COMPREHENSIVE BATCH TESTING")
        logging.info("🚀"*40 + "\n")
        
        start_time = datetime.now()
        
        # Test each trader
        for trader in self.all_traders:
            await self.test_single_trader(trader)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Generate master comparison report
        self._generate_master_report(duration)
        
        logging.info("\n" + "🎉"*40)
        logging.info("BATCH TESTING COMPLETE!")
        logging.info("🎉"*40 + "\n")
    
    def _generate_master_report(self, duration):
        """Generate master comparison report and trader scorecard"""
        
        if not self.master_results:
            logging.warning("No master results to report")
            return
        
        # Create master DataFrame
        master_df = pd.DataFrame(self.master_results)
        
        # Sort by best P&L
        master_df = master_df.sort_values('best_pnl', ascending=False)
        
        # Save master CSV
        master_csv = self.master_output_dir / "MASTER_COMPARISON.csv"
        master_df.to_csv(master_csv, index=False)
        
        # Generate master scorecard report
        report = f"""
{'='*80}
🏆 MASTER TRADER SCORECARD
{'='*80}
Batch Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration}
Mode: {'QUICK' if self.quick_mode else 'FULL'}
Traders Tested: {len(self.all_traders)}

{'='*80}
📊 TRADER RANKINGS (by Best P&L):
{'='*80}

"""
        
        for idx, row in master_df.iterrows():
            rank = master_df.index.get_loc(idx) + 1
            
            report += f"""
{'─'*80}
#{rank}. {row['trader'].upper()}
{'─'*80}
Best P&L:              ${row['best_pnl']:,.2f}
Best win rate:         {row['best_win_rate']:.1f}%
Best profit factor:    {row['best_profit_factor']:.2f}
Profitable configs:    {row['profitable_configs']} / {row['total_tests']} ({row['profitable_configs']/row['total_tests']*100:.1f}%)
Avg trades:            {row['avg_trades']:.0f}

RECOMMENDATION:        {row['recommendation']}

"""
        
        # Add summary statistics
        profitable_traders = len(master_df[master_df['best_pnl'] > 0])
        
        report += f"""
{'='*80}
📈 SUMMARY STATISTICS:
{'='*80}

Total traders tested:     {len(master_df)}
Profitable traders:       {profitable_traders}
Unprofitable traders:     {len(master_df) - profitable_traders}

Best overall P&L:         ${master_df['best_pnl'].max():,.2f} ({master_df.iloc[0]['trader']})
Worst overall P&L:        ${master_df['best_pnl'].min():,.2f} ({master_df.iloc[-1]['trader']})

{'='*80}
💡 RECOMMENDATIONS:
{'='*80}

✅ TRADE THESE TRADERS:
"""
        
        # List traders to trade
        for _, row in master_df[master_df['best_pnl'] > 0].iterrows():
            report += f"   - {row['trader'].upper()} (${row['best_pnl']:,.2f} best P&L)\n"
        
        if profitable_traders == 0:
            report += "   None - all traders tested unprofitably\n"
        
        report += f"""
❌ AVOID THESE TRADERS:
"""
        
        # List traders to avoid
        for _, row in master_df[master_df['best_pnl'] <= 0].iterrows():
            report += f"   - {row['trader'].upper()} (${row['best_pnl']:,.2f} best P&L)\n"
        
        report += f"""
{'='*80}
📁 OUTPUT FILES:
{'='*80}

Master comparison:        {master_csv}
Individual reports:       {self.master_output_dir}/<trader>/<trader>_optimization_summary.txt
Full results CSVs:        {self.master_output_dir}/<trader>/<trader>_all_results.csv

{'='*80}
"""
        
        # Save master report
        report_file = self.master_output_dir / "MASTER_SCORECARD.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Print to console
        print(report)
        
        logging.info(f"✅ Master scorecard saved: {report_file}")
        logging.info(f"✅ Master comparison CSV saved: {master_csv}")


async def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive batch testing for all Discord traders"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (16 tests per trader instead of 3,614)'
    )
    parser.add_argument(
        '--traders',
        nargs='+',
        help='Specific traders to test (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create batch tester
    batch_tester = BatchTesterAllTraders(
        quick_mode=args.quick,
        trader_list=args.traders
    )
    
    # Run batch tests
    await batch_tester.run_batch_tests()


if __name__ == "__main__":
    asyncio.run(main())
