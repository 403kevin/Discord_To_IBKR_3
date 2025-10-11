#!/bin/bash
# setup_backtesting.sh - Prepare folder structure for automated backtesting

echo "🚀 Setting up backtesting automation structure..."

# Create channel signals directory
mkdir -p backtester/channel_signals

# Create sample channel files
for i in {01..20}; do
    file="backtester/channel_signals/channel_${i}_signals.txt"
    if [ ! -f "$file" ]; then
        echo "# Channel $i signals - Add your signals below (one per line)" > "$file"
        echo "# Format: TICKER MM/DD STRIKE[C/P]" >> "$file"
        echo "# Example: SPY 10/15 500C" >> "$file"
        echo "" >> "$file"
    fi
done

# Create optimization results directory
mkdir -p backtester/optimization_results

# Create historical data directory if not exists
mkdir -p backtester/historical_data

echo "✅ Folder structure created!"
echo ""
echo "📂 Created directories:"
echo "   backtester/channel_signals/     - Put your signals here (20 files created)"
echo "   backtester/optimization_results/ - Results will be saved here"
echo "   backtester/historical_data/      - Downloaded options data"
echo ""
echo "📝 Next steps:"
echo "1. Add your signals to channel_XX_signals.txt files"
echo "2. Run: python parameter_optimizer.py --quick  (for single channel test)"
echo "3. Run: python optimize_all_channels.py --quick  (for all channels)"
