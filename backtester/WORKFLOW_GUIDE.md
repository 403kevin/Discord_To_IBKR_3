# 🎯 COMPLETE WORKFLOW: Testing 20 Channels with Parameter Optimization

## 📋 YOUR REQUIREMENTS

✅ **20 channels** to test (one at a time)
✅ **ONE** `signals_to_test.txt` file per run
✅ **MULTIPLE parameter combinations** to find optimal settings

---

## 🗂️ FILE STRUCTURE

```
backtester/
├── channel_signals/              # Create this folder
│   ├── channel_01_signals.txt    # Channel 1 signals
│   ├── channel_02_signals.txt    # Channel 2 signals
│   ├── ...
│   └── channel_20_signals.txt    # Channel 20 signals
│
├── signals_to_test.txt           # Working file (gets overwritten)
├── historical_data/              # Downloaded options data
│
├── optimization_results/         # Results for each channel
│   ├── channel_01/
│   │   ├── all_results.csv       # All parameter combinations
│   │   └── optimization_summary.txt
│   ├── channel_02/
│   └── ...
│
└── parameter_optimizer.py        # Main optimization script
```

---

## ⚡ QUICK START (3 Steps)

### **Step 1: Organize Your Signals**

Create one file per channel:

```bash
mkdir -p backtester/channel_signals

# Copy your signals into files like:
# channel_01_signals.txt - Channel name: "Trader Alpha"
# channel_02_signals.txt - Channel name: "Options Flow"
# etc.
```

**Format in each file:**
```
SPY 10/15 500C
QQQ 10/15 450P
# One signal per line
```

---

### **Step 2: Run Optimization for ONE Channel**

```bash
# Copy one channel's signals to the working file
cp backtester/channel_signals/channel_01_signals.txt backtester/signals_to_test.txt

# Download historical data
python backtester/data_harvester.py

# Run optimization (tests ALL parameter combinations)
python parameter_optimizer.py

# Results appear in: backtester/optimization_results/TIMESTAMP/
```

---

### **Step 3: Repeat for All 20 Channels**

**Option A: Manual** (one at a time)
```bash
# For each channel:
cp backtester/channel_signals/channel_02_signals.txt backtester/signals_to_test.txt
python backtester/data_harvester.py
python parameter_optimizer.py
# Rename results folder: mv backtester/optimization_results/TIMESTAMP backtester/optimization_results/channel_02
```

**Option B: Automated** (all channels overnight)
```bash
python optimize_all_channels.py

# Or quick mode (fewer combinations, faster):
python optimize_all_channels.py --quick
```

---

## 🔬 PARAMETER COMBINATIONS

### **Quick Mode** (16 tests per channel = ~5 minutes)
- Breakeven: [10%, 15%]
- Trail: [atr, pullback]
- PSAR: [On, Off]
- RSI: [On, Off]

**Total:** 20 channels × 16 tests = **320 backtests**

### **Full Mode** (Default - ~3,456 tests per channel = ~2 hours)
All possible combinations of:
- Breakeven: [5%, 10%, 15%, 20%]
- Trail Method: [atr, pullback_percent]
- Pullback: [8%, 10%, 12%, 15%]
- ATR Period: [10, 14, 20]
- ATR Multiplier: [1.0, 1.5, 2.0, 2.5]
- PSAR: [On, Off] with [0.01, 0.02, 0.03] for start/increment
- RSI: [On, Off] with different periods/levels

**Total:** 20 channels × 3,456 tests = **69,120 backtests** (~40 hours)

### **Custom Mode** (You define exactly what to test)
```bash
# Create custom parameter grid
nano param_grid_custom.json

# Run with custom parameters
python parameter_optimizer.py --params param_grid_custom.json
```

**Example Custom Grid:**
```json
{
  "breakeven_trigger_percent": [10, 15],
  "trail_method": ["atr"],
  "atr_multiplier": [1.5, 2.0],
  "psar_enabled": [true, false],
  "rsi_hook_enabled": [true]
}
```
This would test: 2 × 1 × 2 × 2 × 1 = **8 combinations** per channel

---

## 📊 OUTPUT FILES

### **Per-Channel Results:**
```
backtester/optimization_results/channel_01/
├── all_results.csv                # Every parameter combination tested
└── optimization_summary.txt       # Analysis + recommended config
```

**all_results.csv** contains:
- test_name
- total_trades, total_pnl, win_rate
- profit_factor, max_drawdown, sharpe_ratio
- breakeven_trigger_percent
- trail_method, pullback_percent
- atr_period, atr_multiplier
- psar_enabled, psar_start, psar_increment, psar_max
- rsi_hook_enabled, rsi_period, rsi_overbought, rsi_oversold

**optimization_summary.txt** contains:
- Top 10 configs by P&L
- Top 10 by win rate
- Top 10 by profit factor
- Parameter analysis (which settings matter most)
- **RECOMMENDED CONFIGURATION** (copy-paste into config.py)

### **Master Results (All Channels):**
```
backtester/
├── master_optimization_TIMESTAMP.csv    # Best config for each channel
└── master_summary_TIMESTAMP.txt         # Rankings across all channels
```

---

## 🎯 TYPICAL WORKFLOW

### **Phase 1: Quick Discovery** (~2 hours for 20 channels)
```bash
python optimize_all_channels.py --quick
```
- Tests 16 combinations per channel
- Finds generally good parameters
- Identifies which channels are profitable

### **Phase 2: Deep Optimization** (overnight for top 5 channels)
```bash
# For your 5 most profitable channels from Phase 1:
cp backtester/channel_signals/channel_03_signals.txt backtester/signals_to_test.txt
python parameter_optimizer.py  # Full grid
```
- Tests ~3,456 combinations
- Finds absolute best parameters
- Fine-tunes for maximum profit

### **Phase 3: Apply Results**
```bash
# Edit services/config.py
# Copy the "RECOMMENDED CONFIGURATION" from optimization_summary.txt
# Create one profile per profitable channel
```

---

## 📈 EXAMPLE OUTPUT

```
TOP 10 PARAMETER COMBINATIONS (by Total P&L):

#1. test_0042 - P&L: $4,587.50
   Trades: 23 | Win Rate: 65.2% | Profit Factor: 2.34
   Breakeven: 10% | Trail: atr
   ATR: 1.5x | Pullback: 10%
   PSAR: True | RSI Hook: False

#2. test_0156 - P&L: $4,234.00
   ...

PARAMETER ANALYSIS:

Best Breakeven Trigger:
  10%: Avg P&L $2,345.67
  15%: Avg P&L $2,123.45
  5%: Avg P&L $1,987.34
  20%: Avg P&L $1,654.23

Best Trail Method:
  atr: Avg P&L $2,456.78
  pullback_percent: Avg P&L $2,012.34

PSAR Impact:
  Enabled: Avg P&L $2,567.89
  Disabled: Avg P&L $2,234.56

RECOMMENDED CONFIGURATION:
----------------------------------------
"exit_strategy": {
    "breakeven_trigger_percent": 10,
    "trail_method": "atr",
    "trail_settings": {
        "pullback_percent": 10,
        "atr_period": 14,
        "atr_multiplier": 1.5
    },
    "momentum_exits": {
        "psar_enabled": true,
        "psar_settings": {"start": 0.02, "increment": 0.02, "max": 0.2},
        "rsi_hook_enabled": false,
        "rsi_settings": {"period": 14, "overbought_level": 70, "oversold_level": 30}
    }
}
```

---

## 🚀 COMMANDS REFERENCE

### **Single Channel Optimization:**
```bash
# Quick test (16 combinations)
python parameter_optimizer.py --quick

# Full test (3,456 combinations)
python parameter_optimizer.py

# Custom parameters
python parameter_optimizer.py --params my_grid.json
```

### **All Channels:**
```bash
# Quick mode (all 20 channels, 16 tests each)
python optimize_all_channels.py --quick

# Full mode (all 20 channels, 3,456 tests each)
python optimize_all_channels.py
```

### **Custom Workflow:**
```bash
# Test specific signals file
python parameter_optimizer.py --signals backtester/my_signals.txt --quick
```

---

## ⏱️ TIME ESTIMATES

| Mode | Tests/Channel | Time/Channel | 20 Channels Total |
|------|---------------|--------------|-------------------|
| Quick | 16 | ~6 min | ~2 hours |
| Medium | 864 | ~30 min | ~10 hours |
| Full | 3,456 | ~2 hours | ~40 hours |

*Times assume 5-second bar data and ~50 signals per channel*

---

## 💡 PRO TIPS

1. **Start with quick mode** to identify profitable channels
2. **Focus full optimization** on your top 5 channels only
3. **Run overnight** for full optimization (40 hours total)
4. **Custom grids** let you test specific hunches efficiently
5. **Compare win rate vs P&L** - high WR with low P&L means poor R:R
6. **Parameter analysis** tells you which settings actually matter
7. **Profit factor > 1.5** is generally good for day trading
8. **Watch max drawdown** - high P&L with huge drawdown = risky

---

## 🔥 NEXT STEPS

1. **Download files:**
   - `parameter_optimizer.py`
   - `optimize_all_channels.py`
   - `param_grid_template.json`

2. **Create channel folders:**
   ```bash
   mkdir -p backtester/channel_signals
   ```

3. **Add your signals:**
   - One file per channel
   - Name: `channel_01_signals.txt`, `channel_02_signals.txt`, etc.

4. **Run quick test on ONE channel:**
   ```bash
   cp backtester/channel_signals/channel_01_signals.txt backtester/signals_to_test.txt
   python backtester/data_harvester.py
   python parameter_optimizer.py --quick
   ```

5. **Review results** in `backtester/optimization_results/`

6. **Scale up** when satisfied with single-channel test

---

Ready to download the scripts?
