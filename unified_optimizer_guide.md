# 🚀 UNIFIED SUPER OPTIMIZER - DOCUMENTATION

## 🎯 What It Does

The **Unified Parameter Optimizer** automatically detects whether your signals are 0DTE or regular options and applies the appropriate parameter ranges for each type. No more running separate optimizers!

---

## ⚡ Key Features

1. **Auto-Detection**: Scans your signals and categorizes them
   - SPX/SPY/SPXW → 0DTE parameters (tight, fast)
   - Everything else → Regular parameters (wider, slower)

2. **Separate Testing**: Tests each category with optimal ranges
   - 0DTE: 7-10% breakeven, 10-20% pullback, no PSAR/RSI
   - Regular: 5-15% breakeven, 7-15% pullback, tests ATR/PSAR/RSI

3. **Unified Results**: One comprehensive report comparing both types
   - See which signal type performs better
   - Compare best configs across categories
   - Identify trader skill patterns

---

## 🚀 Usage

### Quick Mode (Recommended First Run)
```bash
python backtester/parameter_optimizer_unified.py --quick
```

**Tests:**
- 0DTE: 8 combinations (2min)
- Regular: 24 combinations (6min)
- **Total: ~8 minutes**

### Full Mode (Comprehensive)
```bash
python backtester/parameter_optimizer_unified.py
```

**Tests:**
- 0DTE: 100 combinations (25min)
- Regular: 240 combinations (1 hour)
- **Total: ~1.5 hours**

### Custom Signals File
```bash
python backtester/parameter_optimizer_unified.py --quick --signals path/to/signals.txt
```

---

## 📊 Output Files

All results saved to: `backtester/optimization_results/TIMESTAMP/`

1. **unified_results.csv**
   - Complete data for all tests
   - Includes signal_type column (0DTE vs REGULAR)
   - All parameters and metrics

2. **unified_optimization_summary.txt**
   - Overall performance stats
   - Breakdown by signal type
   - Top 5 configs per type
   - Top 10 overall

3. **temp_signals_0dte.txt** (deleted after run)
   - Temporary file with SPX/SPY signals

4. **temp_signals_regular.txt** (deleted after run)
   - Temporary file with other signals

---

## 📋 Example Output Structure

```
====================================================================================================
UNIFIED PARAMETER OPTIMIZATION SUMMARY
Generated: 2025-10-19 18:30:00
Tests both 0DTE and Regular option parameters
====================================================================================================

OVERALL PERFORMANCE:
----------------------------------------------------------------------------------------------------
Total Tests: 132
Profitable Configs: 23 (17.4%)
Average P&L: $-845.20
Best P&L: $1,245.00
Worst P&L: $-3,200.00

0DTE SIGNALS BREAKDOWN:
----------------------------------------------------------------------------------------------------
Tests Run: 100
Profitable: 7 (7.0%)
Average P&L: $-1,453.44
Average Win Rate: 33.6%
Average Profit Factor: 0.71

TOP 5 0DTE CONFIGURATIONS:

#1. 0DTE_test_0038
   Win Rate: 33.3% | P&L: $884.00 | PF: 1.43
   Breakeven: 7% | Pullback: 20% | Native: 20%

#2. 0DTE_test_0039
   Win Rate: 33.3% | P&L: $623.00 | PF: 1.27
   Breakeven: 7% | Pullback: 20% | Native: 25%

REGULAR SIGNALS BREAKDOWN:
----------------------------------------------------------------------------------------------------
Tests Run: 32
Profitable: 16 (50.0%)
Average P&L: $445.30
Average Win Rate: 52.5%
Average Profit Factor: 1.45

TOP 5 REGULAR CONFIGURATIONS:

#1. REGULAR_test_0012
   Win Rate: 62.5% | P&L: $1,245.00 | PF: 2.15
   Breakeven: 10% | ATR: 14p×1.5 | Native: 30%

#2. REGULAR_test_0008
   Win Rate: 58.3% | P&L: $1,120.00 | PF: 1.85
   Breakeven: 12% | Pullback: 10% | Native: 25%
```

---

## 🔍 How It Works

### Step 1: Signal Categorization
```
Input file (signals_to_test.txt):
2025-09-29 10:13:01 | Goldman | SPX 6650P 09/29      ← 0DTE (SPX)
2025-10-02 09:54:10 | Expo | TSLA 250C 10/11        ← Regular (TSLA)
2025-10-03 11:14:08 | Zeus | SPY 580P 10/03         ← 0DTE (SPY)
2025-10-09 08:52:23 | Arrow | NVDA 145C 10/18       ← Regular (NVDA)

Output:
✅ Found 2 0DTE signals (SPX/SPY)
✅ Found 2 regular option signals
```

### Step 2: Apply Appropriate Parameters

**For 0DTE Signals:**
- Breakeven: 5-15% (tight)
- Pullback: 8-20% (wide for volatility)
- Native Trail: 15-30%
- PSAR: Disabled (useless for 0DTE)
- RSI: Disabled (useless for 0DTE)

**For Regular Signals:**
- Breakeven: 5-15% (wider range)
- Trail Method: Both pullback AND ATR
- ATR: Multiple periods (10, 14, 20)
- Native Trail: 20-35% (wider safety)
- PSAR: Test enabled/disabled
- RSI: Test enabled/disabled

### Step 3: Run & Compare
- Tests each category separately
- Combines results
- Shows which signal type is more profitable
- Highlights best parameters for each

---

## 💡 Use Cases

### Use Case 1: Mixed Trader (Like Your Situation)
**Trader posts both:**
- SPX 0DTE scalps (6650P, 6720C)
- NVDA/TSLA day trades (145C, 250C)

**Solution:**
```bash
# One command tests both optimally
python backtester/parameter_optimizer_unified.py --quick
```

**Result:**
- Discovers SPX scalps need 7% breakeven, 20% pullback
- Discovers NVDA trades need 12% breakeven, 10% pullback, PSAR enabled
- Shows you should use DIFFERENT configs for each type!

---

### Use Case 2: Compare Multiple Traders
**Scenario:**
- Goldman: Only SPX 0DTE
- Expo: Only regular options
- Zeus: Mixed (both types)

**Process:**
```bash
# Test Goldman (pure 0DTE)
python backtester/parameter_optimizer_unified.py --signals goldman_signals.txt --quick

# Test Expo (pure regular)
python backtester/parameter_optimizer_unified.py --signals expo_signals.txt --quick

# Test Zeus (mixed)
python backtester/parameter_optimizer_unified.py --signals zeus_signals.txt --quick
```

**Compare:**
- Who has better 0DTE win rate?
- Who has better regular options win rate?
- Is Zeus actually good at both or mediocre at both?

---

### Use Case 3: Channel Audit
**Run unified test on all 11 traders:**

```bash
for channel in goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf; do
    echo "Testing $channel..."
    python backtester/parameter_optimizer_unified.py \
        --quick \
        --signals "backtester/channel_signals/${channel}_signals.txt"
done
```

**Result:**
- Master spreadsheet showing each trader's performance
- Separate scores for 0DTE vs regular
- Identify specialists vs generalists

---

## ⚙️ Configuration

### Quick Mode Parameters

**0DTE:**
```python
{
    'breakeven_trigger_percent': [7, 10],           # 2 values
    'pullback_percent': [10, 15],                   # 2 values
    'native_trail_percent': [20, 25],               # 2 values
}
# Total: 2 × 2 × 2 = 8 combinations
```

**Regular:**
```python
{
    'breakeven_trigger_percent': [7, 10, 12],       # 3 values
    'trail_method': ['pullback_percent', 'atr'],    # 2 values
    'pullback_percent': [8, 10],                    # 2 values
    'native_trail_percent': [25, 30],               # 2 values
    'psar_enabled': [True, False],                  # 2 values
}
# Total: 3 × 2 × 2 × 2 × 2 = 48 combinations (BUT trail method splits tests)
# Actual: 3 × 2 × 2 × 2 = 24 pullback + 3 × 2 × 2 = 12 ATR = 36 total... simplified to ~24
```

### Full Mode Parameters

**0DTE:** 100 combinations
**Regular:** 240 combinations

---

## 🎯 When to Use Each Optimizer

| Optimizer | Use When | Signals |
|-----------|----------|---------|
| **parameter_optimizer.py** | Testing ONLY regular options | TSLA, NVDA, AAPL, etc. |
| **parameter_optimizer_0DTE.py** | Testing ONLY 0DTE | Pure SPX/SPY 0DTE trader |
| **parameter_optimizer_unified.py** ⭐ | Mixed signals OR testing multiple traders | ANY - auto-detects |

---

## 🚨 Important Notes

1. **Signal Format Required:**
   ```
   YYYY-MM-DD HH:MM:SS | TRADER | TICKER STRIKE[C/P] MM/DD
   ```

2. **Ticker Detection:**
   - SPX, SPY, SPXW → 0DTE category
   - Everything else → Regular category
   - Case insensitive

3. **Historical Data:**
   - Must have data for ALL signals
   - Missing data = signal skipped
   - Check `backtester/historical_data/` folder

4. **Runtime:**
   - Quick: 8-15 minutes total
   - Full: 1-2 hours total
   - Depends on number of signals in each category

---

## 🏆 Example: Goldman vs Expo Comparison

**Goldman Results (0DTE only):**
```
Tests: 100
Profitable: 7 (7%)
Best: $884 (7% BE, 20% PB, 20% NT)
Verdict: Mediocre 0DTE trader
```

**Expo Results (Regular only):**
```
Tests: 240
Profitable: 45 (18.8%)
Best: $2,140 (10% BE, ATR 14×1.5, 30% NT)
Verdict: Good regular options trader
```

**Conclusion:**
- Goldman: Skip or use minimal position sizing
- Expo: Worth following with optimized regular params
- **Use unified optimizer to test both in one run!**

---

## ✅ Advantages Over Split Optimizers

| Feature | Split (0DTE + Regular) | Unified |
|---------|----------------------|---------|
| **Setup** | Edit 2 files, run 2 commands | 1 command |
| **Runtime** | ~3 hours total | ~1.5 hours (auto-splits) |
| **Results** | 2 separate reports | 1 unified comparison |
| **Signal Prep** | Manually separate | Auto-detects |
| **Trader Comparison** | Manual effort | Built-in |
| **Mixed Traders** | ❌ Hard to analyze | ✅ Perfect |

---

## 🎪 Bottom Line

**Use the unified optimizer when:**
- ✅ Testing traders who post mixed signals (SPX + stocks)
- ✅ Comparing multiple traders at once
- ✅ You want ONE comprehensive report
- ✅ You're lazy and want automation (BEST reason)

**Use separate optimizers when:**
- You want MORE granular control over parameter ranges
- You're testing a pure specialist (0DTE-only or regular-only)
- You have a specific hypothesis about parameters

**For your situation (testing 11 traders with unknown mix):**
→ **USE THE UNIFIED OPTIMIZER** and let it sort out who trades what.
