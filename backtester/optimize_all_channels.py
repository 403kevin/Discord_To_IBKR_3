#!/usr/bin/env python3
"""
optimize_all_channels.py - Automated Multi-Channel Optimization Pipeline
=========================================================================
This script automates testing across all your Discord channels:
1. Processes each channel's signals one by one
2. Runs parameter optimization for each
3. Generates master comparison report
4. Identifies the most profitable channels and configurations

Usage:
    python optimize_all_channels.py                  # Full optimization
    python optimize_all_channels.py --quick          # Quick mode
    python optimize_all_channels.py --channels 5     # Test only first 5 channels
"""

import asyncio
import logging
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.data_harvester import DataHarvester
from backtester.parameter_optimizer import ParameterOptimizer
from services.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class MultiChannelOptimizer:
    """
    Orchestrates optimization across multiple channels.
    The ultimate automation - set it and forget it!
    """
    
    def __init__(self, channels_dir="backtester/channel_signals", quick_mode=False, max_channels=None):
        self.channels_dir = Path(channels_dir)
        self.quick_mode = quick_mode
        self.max_channels = max_channels
        self.master_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.master_output_dir = Path(f"backtester/optimization_results/master_{self.timestamp}")
        self.master_output_dir.mkdir(exist_ok=True, parents=True)
        
    async def run_all_channels(self):
        """Process all channel signal files automatically."""
        # Find all signal files
        signal_files = sorted(self.channels_dir.glob("channel_*_signals.txt"))
        
        if not signal_files:
            logging.error(f"No channel signal files found in {self.channels_dir}")
            logging.info("\nExpected format:")
            logging.info("  channel_01_signals.txt")
            logging.info("  channel_02_signals.txt")
            logging.info("  ... etc")
            logging.info("\nCreate these files with your signals (one signal per line)")
            return
        
        # Limit channels if requested
        if self.max_channels:
            signal_files = signal_files[:self.max_channels]
        
        logging.info(f"\n🎯 Found {len(signal_files)} channels to optimize")
        logging.info(f"Mode: {'Quick' if self.quick_mode else 'Full'}")
        
        # Estimate total time
        tests_per_channel = 16 if self.quick_mode else 3456
        time_per_test = 5  # seconds
        total_time_hours = (len(signal_files) * tests_per_channel * time_per_test) / 3600
        logging.info(f"Estimated total time: ~{total_time_hours:.1f} hours\n")
        
        # Process each channel
        for i, signal_file in enumerate(signal_files, 1):
            channel_name = signal_file.stem.replace("_signals", "")
            logging.info(f"\n{'='*100}")
            logging.info(f"CHANNEL {i}/{len(signal_files)}: {channel_name}")
            logging.info(f"{'='*100}")
            
            try:
                await self.process_single_channel(channel_name, signal_file, i)
            except Exception as e:
                logging.error(f"Failed to process {channel_name}: {e}")
                continue
        
        # Generate master report
        self.generate_master_report()
    
    async def process_single_channel(self, channel_name: str, signal_file: Path, channel_num: int):
        """Process optimization for one channel."""
        test_signals_file = Path("backtester/signals_to_test.txt")
        
        try:
            # Step 1: Copy signals to working file
            logging.info(f"Step 1/4: Preparing signals from {signal_file.name}")
            shutil.copy(signal_file, test_signals_file)
            
            # Count signals
            with open(test_signals_file, 'r') as f:
                signal_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
            logging.info(f"  Found {signal_count} signals")
            
            # Step 2: Harvest historical data
            logging.info(f"Step 2/4: Downloading historical options data")
            harvester = DataHarvester()
            
            # Check if data already exists (to save time)
            data_dir = Path("backtester/historical_data")
            existing_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
            
            if len(existing_files) >= signal_count * 0.8:  # If we have 80% of data
                logging.info(f"  Using existing data ({len(existing_files)} files)")
            else:
                logging.info(f"  Downloading fresh data from IBKR...")
                await harvester.run()
            
            # Step 3: Run parameter optimization
            logging.info(f"Step 3/4: Running parameter optimization")
            logging.info(f"  Testing {16 if self.quick_mode else 3456} parameter combinations")
            
            optimizer = ParameterOptimizer(
                signals_file=str(test_signals_file),
                quick_mode=self.quick_mode
            )
            
            await optimizer.run_optimization()
            
            # Step 4: Save channel-specific results
            logging.info(f"Step 4/4: Saving results for {channel_name}")
            
            if optimizer.results:
                # Find the output directory that was just created
                latest_dir = max(
                    Path("backtester/optimization_results").glob("20*"),
                    key=lambda p: p.stat().st_mtime
                )
                
                # Move to channel-specific folder
                channel_dir = self.master_output_dir / channel_name
                if channel_dir.exists():
                    shutil.rmtree(channel_dir)
                shutil.move(str(latest_dir), str(channel_dir))
                
                # Track best result for master report
                best_result = max(optimizer.results, key=lambda x: x['total_pnl'])
                best_result['channel'] = channel_name
                best_result['channel_num'] = channel_num
                best_result['signal_count'] = signal_count
                self.master_results.append(best_result)
                
                logging.info(f"✅ {channel_name} complete!")
                logging.info(f"   Best P&L: ${best_result['total_pnl']:.2f}")
                logging.info(f"   Win Rate: {best_result['win_rate']:.1f}%")
                logging.info(f"   Profit Factor: {best_result['profit_factor']:.2f}")
            else:
                logging.warning(f"No results generated for {channel_name}")
            
        except Exception as e:
            logging.error(f"Error processing {channel_name}: {e}", exc_info=True)
    
    def generate_master_report(self):
        """Create master comparison report across all channels."""
        if not self.master_results:
            logging.warning("No results to compile")
            return
        
        logging.info(f"\n{'='*100}")
        logging.info("GENERATING MASTER REPORT")
        logging.info(f"{'='*100}")
        
        # Create DataFrame
        df = pd.DataFrame(self.master_results)
        
        # Save full comparison
        master_csv = self.master_output_dir / "master_comparison.csv"
        df.to_csv(master_csv, index=False)
        
        # Generate summary report
        summary_file = self.master_output_dir / "MASTER_SUMMARY.txt"
        with open(summary_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("MASTER OPTIMIZATION SUMMARY - ALL CHANNELS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Channels Tested: {len(self.master_results)}\n")
            f.write(f"Mode: {'Quick' if self.quick_mode else 'Full'}\n")
            f.write("="*100 + "\n\n")
            
            # Top channels by P&L
            f.write("🏆 TOP CHANNELS BY PROFITABILITY:\n")
            f.write("-"*100 + "\n")
            df_sorted = df.sort_values('total_pnl', ascending=False)
            
            for i, row in df_sorted.head(10).iterrows():
                f.write(f"\n#{i+1}. {row['channel']} - Total P&L: ${row['total_pnl']:.2f}\n")
                f.write(f"   Signals: {row['signal_count']} | Trades: {row['total_trades']} | ")
                f.write(f"Win Rate: {row['win_rate']:.1f}% | PF: {row['profit_factor']:.2f}\n")
                f.write(f"   Best Config: Breakeven {row['breakeven_trigger_percent']}% | ")
                f.write(f"Trail: {row['trail_method']}")
                if row['trail_method'] == 'atr':
                    f.write(f" ({row['atr_multiplier']}x)")
                else:
                    f.write(f" ({row['pullback_percent']}%)")
                f.write(f" | PSAR: {row['psar_enabled']} | RSI: {row['rsi_hook_enabled']}\n")
            
            # Statistics
            f.write("\n" + "="*100 + "\n")
            f.write("📊 OVERALL STATISTICS:\n")
            f.write("-"*100 + "\n")
            
            profitable_channels = df[df['total_pnl'] > 0]
            f.write(f"Profitable Channels: {len(profitable_channels)}/{len(df)} ")
            f.write(f"({len(profitable_channels)/len(df)*100:.1f}%)\n")
            f.write(f"Average P&L per Channel: ${df['total_pnl'].mean():.2f}\n")
            f.write(f"Best Channel P&L: ${df['total_pnl'].max():.2f}\n")
            f.write(f"Worst Channel P&L: ${df['total_pnl'].min():.2f}\n")
            f.write(f"Total P&L (All Channels): ${df['total_pnl'].sum():.2f}\n")
            
            # Parameter insights
            f.write("\n" + "="*100 + "\n")
            f.write("🔍 PARAMETER INSIGHTS (Across All Channels):\n")
            f.write("-"*100 + "\n")
            
            # Most common successful parameters
            top_5 = df_sorted.head(5)
            
            f.write(f"\nMost Common Settings in Top 5 Channels:\n")
            f.write(f"  Trail Method: {top_5['trail_method'].mode()[0]}\n")
            f.write(f"  Avg Breakeven: {top_5['breakeven_trigger_percent'].mean():.1f}%\n")
            f.write(f"  PSAR Enabled: {top_5['psar_enabled'].sum()}/{len(top_5)}\n")
            f.write(f"  RSI Enabled: {top_5['rsi_hook_enabled'].sum()}/{len(top_5)}\n")
            
            # Recommended next steps
            f.write("\n" + "="*100 + "\n")
            f.write("📝 RECOMMENDED NEXT STEPS:\n")
            f.write("-"*100 + "\n")
            f.write("\n1. FOCUS ON TOP PERFORMERS:\n")
            
            for i, row in df_sorted.head(3).iterrows():
                f.write(f"   - {row['channel']}: Configure in services/config.py\n")
            
            f.write("\n2. DISABLE POOR PERFORMERS:\n")
            worst_3 = df_sorted.tail(3)
            for i, row in worst_3.iterrows():
                if row['total_pnl'] < 0:
                    f.write(f"   - {row['channel']}: P&L ${row['total_pnl']:.2f}\n")
            
            f.write("\n3. APPLY CONFIGURATIONS:\n")
            f.write("   Copy the exit_strategy from each channel's optimization_summary.txt\n")
            f.write("   to the corresponding profile in services/config.py\n")
            
            f.write("\n4. PAPER TRADE TEST:\n")
            f.write("   Run the bot in paper trading mode with optimized parameters\n")
            f.write("   Monitor for 1 week before going live\n")
            
            f.write("\n" + "="*100 + "\n")
        
        # Print summary to console
        with open(summary_file, 'r') as f:
            print(f.read())
        
        # Save config template
        self.generate_config_template(df_sorted.head(5))
        
        logging.info(f"\n✅ Master results saved to: {master_csv}")
        logging.info(f"✅ Master summary saved to: {summary_file}")
        logging.info(f"✅ All results in: {self.master_output_dir}\n")
    
    def generate_config_template(self, top_channels):
        """Generate a ready-to-use config template for top channels."""
        config_file = self.master_output_dir / "config_template.py"
        
        with open(config_file, 'w') as f:
            f.write("# AUTO-GENERATED CONFIG TEMPLATE\n")
            f.write("# Copy these profiles to your services/config.py\n\n")
            f.write("profiles = [\n")
            
            for i, row in top_channels.iterrows():
                f.write(f"    # {row['channel']} - P&L: ${row['total_pnl']:.2f}\n")
                f.write("    {\n")
                f.write(f'        "channel_id": "YOUR_CHANNEL_ID_HERE",\n')
                f.write(f'        "channel_name": "{row["channel"]}",\n')
                f.write(f'        "enabled": True,\n')
                f.write(f'        "trading": {{\n')
                f.write(f'            "funds_allocation": 1000,\n')
                f.write(f'            "min_price_per_contract": 0.30,\n')
                f.write(f'            "max_price_per_contract": 10.0\n')
                f.write(f'        }},\n')
                f.write(f'        "exit_strategy": {{\n')
                f.write(f'            "breakeven_trigger_percent": {row["breakeven_trigger_percent"]/100:.2f},\n')
                f.write(f'            "trail_method": "{row["trail_method"]}",\n')
                f.write(f'            "trail_settings": {{\n')
                f.write(f'                "pullback_percent": {row["pullback_percent"]/100:.2f},\n')
                f.write(f'                "atr_period": {int(row["atr_period"])},\n')
                f.write(f'                "atr_multiplier": {row["atr_multiplier"]}\n')
                f.write(f'            }},\n')
                f.write(f'            "momentum_exits": {{\n')
                f.write(f'                "psar_enabled": {str(row["psar_enabled"]).lower()},\n')
                f.write(f'                "rsi_hook_enabled": {str(row["rsi_hook_enabled"]).lower()}\n')
                f.write(f'            }}\n')
                f.write(f'        }}\n')
                f.write("    },\n\n")
            
            f.write("]\n")
        
        logging.info(f"✅ Config template saved to: {config_file}")


async def main():
    """Main entry point with CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated optimization for all Discord channels"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (16 tests per channel instead of 3,456)'
    )
    parser.add_argument(
        '--channels',
        type=int,
        help='Number of channels to test (default: all)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='backtester/channel_signals',
        help='Directory containing channel signal files'
    )
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = MultiChannelOptimizer(
        channels_dir=args.dir,
        quick_mode=args.quick,
        max_channels=args.channels
    )
    
    # Run optimization
    await optimizer.run_all_channels()


if __name__ == "__main__":
    asyncio.run(main())
