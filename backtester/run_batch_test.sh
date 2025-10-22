#!/bin/bash
#
# run_batch_test.sh - Easy wrapper for comprehensive batch testing
# =================================================================
# 
# Usage:
#   ./run_batch_test.sh                    # Full mode (all 3,614 tests × 11 traders = ~22 hours)
#   ./run_batch_test.sh --quick            # Quick mode (16 tests × 11 traders = ~2 hours)
#   ./run_batch_test.sh --traders goldman expo qiqo  # Test specific traders only
#

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 COMPREHENSIVE BATCH TESTING SYSTEM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if we're in the right directory
if [ ! -d "backtester" ]; then
    echo "❌ ERROR: Must run from Discord_To_IBKR_3 root directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected: /path/to/Discord_To_IBKR_3"
    exit 1
fi

# Check if batch_test_all_traders.py exists
if [ ! -f "backtester/batch_test_all_traders.py" ]; then
    echo "❌ ERROR: backtester/batch_test_all_traders.py not found"
    echo "   Please ensure the script is in the correct location"
    exit 1
fi

# Check if channel_signals directory exists
if [ ! -d "backtester/channel_signals" ]; then
    echo "❌ ERROR: backtester/channel_signals/ directory not found"
    echo "   Please create this directory and add your trader signal files"
    exit 1
fi

# Count signal files
signal_count=$(ls backtester/channel_signals/*_signals.txt 2>/dev/null | wc -l)
if [ $signal_count -eq 0 ]; then
    echo "⚠️  WARNING: No signal files found in backtester/channel_signals/"
    echo "   Expected files like: goldman_signals.txt, expo_signals.txt, etc."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Found $signal_count trader signal files"
fi

# Display mode
if [[ "$*" == *"--quick"* ]]; then
    echo "📊 Mode: QUICK (16 tests per trader)"
    echo "⏱️  Estimated time: ~10 minutes per trader"
else
    echo "📊 Mode: FULL (3,614 tests per trader)"
    echo "⏱️  Estimated time: ~2 hours per trader"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Starting batch test..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the Python script with all arguments passed through
python backtester/batch_test_all_traders.py "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ BATCH TESTING COMPLETE!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📁 Results saved to: backtester/optimization_results/BATCH_TEST_*/"
    echo ""
    echo "📊 View master scorecard:"
    echo "   cat backtester/optimization_results/BATCH_TEST_*/MASTER_SCORECARD.txt"
    echo ""
    echo "📈 View individual trader reports:"
    echo "   ls backtester/optimization_results/BATCH_TEST_*/*/"
    echo ""
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ BATCH TESTING FAILED!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Exit code: $exit_code"
    echo "   Check the error messages above for details"
    echo ""
    exit $exit_code
fi
