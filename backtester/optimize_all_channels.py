#!/usr/bin/env python3
"""
optimize_all_channels.py - Automate optimization for all 20 channels

Usage:
    python optimize_all_channels.py
    
This will:
1. Loop through all channel signal files
2. Run data harvester for each
3. Run parameter optimization
4. Save results with channel name
5. Generate master comparison report
"""

import asyncio
import logging
import sys
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.data_harvester import DataHarvester
from parameter_optimizer import ParameterOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MultiChannelOptimizer:
    """Runs optimization across all 20 channels automatically."""
    
    def __init__(self, channels_dir="backtester/channel_signals", quick_mode=False):
        self.channels_dir = Path(channels_dir)
        self.quick_mode = quick_mode
        self.master_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def run_all_channels(self):
        """Processes all channel signal files."""
        # Find all signal files
        signal_files = sorted(self.channels_dir.glob("channel_*_signals.txt"))
        
        if not signal_files:
            logging.error(f"No channel signal files found in {self.channels_dir}")
            logging.info("Expected format: channel_01_signals.txt, channel_02_signals.txt, etc.")
            return
        
        logging.info(f"Found {len(signal_files)} channels to optimize")
        
        for i, signal_file in enumerate(signal_files, 1):
            channel_name = signal_file.stem.replace("_signals", "")
            logging.info(f"\n{'='*80}")
            logging.info(f"[{i}/{len(signal_files)}] Processing: {channel_name}")
            logging.info(f"{'='*80}")
            
            await self.process_single_channel(channel_name, signal_file)
        
        # Generate master report
        self.generate_master_report()
    
    async def process_single_channel(self, channel_name, signal_file):
        """Processes optimization for one channel."""
        test_signals_file = Path("backtester/signals_to_test.txt")
        
        try:
            # Step 1: Copy signals
            logging.info(f"Step 1/3: Copying signals from {signal_file}")
            shutil.copy(signal_file, test_signals_file)
            
            # Step 2: Harvest data
            logging.info(f"Step 2/3: Harvesting historical data")
            harvester = DataHarvester(
                signals_path=str(test_signals_file),
                output_dir="backtester/historical_data"
            )
            await harvester.run()
            
            # Step 3: Run optimization
            logging.info(f"Step 3/3: Running parameter optimization")
            optimizer = ParameterOptimizer(
                signals_file=str(test_signals_file),
                quick_mode=self.quick_mode
            )
            optimizer.run_optimization()
            
            # Save results with channel name
            if optimizer.results:
                latest_dir = max(Path("backtester/optimization_results").glob("*"), key=lambda p: p.stat().st_mtime)
                channel_results_dir = Path(f"backtester/optimization_results/{channel_name}")
                
                if channel_results_dir.exists():
                    shutil.rmtree(channel_results_dir)
                
                shutil.move(str(latest_dir), str(channel_results_dir))
                
                # Track best result
                best_result = max(optimizer.results, key=lambda x: x['total_pnl'])
                best_result['channel'] = channel_name
                self.master_results.append(best_result)
                
                logging.info(f"✅ {channel_name} complete. Best P&L: ${best_result['total_pnl']:.2f}")
            
        except Exception as e:
            logging.error(f"Failed to process {channel_name}: {e}", exc_info=True)
    
    def generate_master_report(self):
        """Creates master comparison report across all channels."""
        if not self.master_results:
            logging.error("No results to report")
            return
        
        import pandas as pd
        df = pd.DataFrame(self.master_results)
        df = df.sort_values('total_pnl', ascending=False)
        
        # Save master results
        master_file = Path(f"backtester/master_optimization_{self.timestamp}.csv")
        df.to_csv(master_file, index=False)
        
        # Create summary
        summary_file = Path(f"backtester/master_summary_{self.timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("MASTER OPTIMIZATION SUMMARY - ALL 20 CHANNELS\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*100 + "\n\n")
            
            f.write("CHANNELS RANKED BY PROFITABILITY:\n")
            f.write("-"*100 + "\n")
            for i, row in df.iterrows():
                f.write(f"{row['channel']}: ${row['total_pnl']:.2f} | {row['win_rate']:.1f}% WR | {row['total_trades']} trades\n")
                f.write(f"  Best Config: Breakeven={row['breakeven_trigger_percent']}% | Trail={row['trail_method']} | ")
                f.write(f"ATR={row['atr_multiplier']}x | PSAR={row['psar_enabled']} | RSI={row['rsi_hook_enabled']}\n\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("TOP 5 MOST PROFITABLE CHANNELS:\n")
            f.write("-"*100 + "\n")
            for i, row in df.head(5).iterrows():
                f.write(f"{i+1}. {row['channel']}: ${row['total_pnl']:.2f}\n")
            
            f.write("\n" + "="*100 + "\n")
            f.write("TOP 5 BY WIN RATE:\n")
            f.write("-"*100 + "\n")
            df_by_wr = df.sort_values('win_rate', ascending=False)
            for i, row in df_by_wr.head(5).iterrows():
                f.write(f"{i+1}. {row['channel']}: {row['win_rate']:.1f}%\n")
            
            f.write("\n" + "="*100 + "\n")
        
        print("\n" + open(summary_file).read())
        print(f"\n✅ Master results: {master_file}")
        print(f"✅ Master summary: {summary_file}\n")


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize all 20 channels automatically")
    parser.add_argument('--quick', action='store_true', help='Use quick mode (16 tests per channel)')
    parser.add_argument('--channels', type=str, default='backtester/channel_signals', help='Channel signals directory')
    
    args = parser.parse_args()
    
    optimizer = MultiChannelOptimizer(
        channels_dir=args.channels,
        quick_mode=args.quick
    )
    
    await optimizer.run_all_channels()


if __name__ == "__main__":
    asyncio.run(main())
