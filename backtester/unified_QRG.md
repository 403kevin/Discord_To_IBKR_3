# 🚀 UNIFIED OPTIMIZER - QUICK REFERENCE

## ✅ YES, IT HAS QUICK TEST MODE

The unified optimizer supports **3 modes**:

---

## 📋 THREE USAGE MODES

### 1️⃣ **QUICK MODE** (Built-in - RECOMMENDED) ⚡
```bash
python backtester/parameter_optimizer_unified.py --quick
```

**What it does:**
- 0DTE signals: Tests **8 combinations** (~2 min)
- Regular signals: Tests **24 combinations** (~6 min)
- **Total: 8-10 minutes for mixed signals**

**Quick mode parameters:**
```
0DTE:
  - Breakeven: [7%, 10%]
  - Pullback: [10%, 15%]
  - Native Trail: [20%, 25%]
  = 2 × 2 × 2 = 8 tests

Regular:
  - Breakeven: [7%, 10%, 12%]
  - Trail Method: [pullback, ATR]
  - Pullback: [8%, 10%]
  - Native Trail: [25%, 30%]
  - PSAR: [True, False]
  = 3 × 2 × 2 × 2 × 2 = ~24 tests (simplified)
```

---

### 2️⃣ **FULL MODE** (Built-in - Comprehensive)
```bash
python backtester/parameter_optimizer_unified.py
```

**What it does:**
- 0DTE signals: Tests **100 combinations** (~25 min)
- Regular signals: Tests **240 combinations** (~1 hour)
- **Total: 1-2 hours for mixed signals**

**Full mode parameters:**
```
0DTE:
  - Breakeven: [5%, 7%, 10%, 12%, 15%]
  - Pullback: [8%, 10%, 12%, 15%, 20%]
  - Native Trail: [15%, 20%, 25%, 30%]
  = 5 × 5 × 4 = 100 tests

Regular:
  - Breakeven: [5%, 7%, 10%, 12%, 15%]
  - Trail Method: [pullback, ATR]
  - Pullback: [7%, 10%, 12%, 15%]
  - ATR: [10p, 14p, 20p] × [1.0, 1.5, 2.0]
  - Native Trail: [20%, 25%, 30%, 35%]
  - PSAR: [True, False]
  - RSI: [True, False]
  = ~240 tests (complex combinations)
```

---

### 3️⃣ **CUSTOM MODE** (Your own parameter grids)
```bash
python backtester/parameter_optimizer_unified.py --params param_grid_unified.json
```

**What it does:**
- Uses YOUR custom parameter ranges from JSON file
- Separate grids for 0DTE and regular
- Full control over what gets tested

**File: param_grid_unified.json**
```json
{
  "0dte_params": {
    "breakeven_trigger_percent": [7, 10],
    "pullback_percent": [10, 15, 20],
    "native_trail_percent": [20, 25]
  },
  "regular_params": {
    "breakeven_trigger_percent": [7, 10, 12],
    "trail_method": ["pullback_percent", "atr"],
    "pullback_percent": [8, 10, 12],
    "native_trail_percent": [25, 30],
    "psar_enabled": [true, false]
  }
}
```

---

## 🎯 WHICH MODE TO USE?

### Use **Quick Mode** when:
- ✅ First time testing a trader
- ✅ Want fast results (8-10 min)
- ✅ Exploring multiple traders quickly
- ✅ Initial parameter discovery

```bash
python backtester/parameter_optimizer_unified.py --quick
```

---

### Use **Full Mode** when:
- ✅ Found a profitable trader
- ✅ Want comprehensive optimization
- ✅ Need to test all parameter ranges
- ✅ Have 1-2 hours to spare

```bash
python backtester/parameter_optimizer_unified.py
```

---

### Use **Custom Mode** when:
- ✅ Quick/full mode showed promising ranges
- ✅ Want to narrow down specific parameters
- ✅ Testing a specific hypothesis
- ✅ Need fine-tuned control

```bash
python backtester/parameter_optimizer_unified.py --params param_grid_unified.json
```

---

## 📊 COMPLETE USAGE EXAMPLES

### Example 1: Quick test on default signals
```bash
python backtester/parameter_optimizer_unified.py --quick
```

---

### Example 2: Full test on specific trader
```bash
python backtester/parameter_optimizer_unified.py \
    --signals backtester/channel_signals/expo_signals.txt
```

---

### Example 3: Quick test all 11 traders
```bash
for trader in goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $trader"
    python backtester/parameter_optimizer_unified.py \
        --quick \
        --signals "backtester/channel_signals/${trader}_signals.txt"
done
```

---

### Example 4: Custom parameters for Goldman (after finding he's trash)
Create `goldman_custom.json`:
```json
{
  "0dte_params": {
    "breakeven_trigger_percent": [5, 7],
    "pullback_percent": [15, 20, 25],
    "native_trail_percent": [15, 20]
  }
}
```

Run:
```bash
python backtester/parameter_optimizer_unified.py \
    --params goldman_custom.json \
    --signals backtester/channel_signals/goldman_signals.txt
```

---

## 🔧 CUSTOM PARAMETER GRID FORMAT

The unified optimizer uses **TWO separate grids**:

```json
{
  "0dte_params": {
    // Parameters for SPX/SPY 0DTE signals
  },
  "regular_params": {
    // Parameters for stock option signals
  }
}
```

### Minimal custom grid:
```json
{
  "0dte_params": {
    "breakeven_trigger_percent": [7],
    "trail_method": ["pullback_percent"],
    "pullback_percent": [20],
    "native_trail_percent": [20],
    "psar_enabled": [false],
    "rsi_hook_enabled": [false]
  },
  "regular_params": {
    "breakeven_trigger_percent": [10],
    "trail_method": ["pullback_percent"],
    "pullback_percent": [10],
    "native_trail_percent": [30],
    "psar_enabled": [false],
    "rsi_hook_enabled": [false]
  }
}
```
**Result:** Tests 1 config for 0DTE, 1 config for regular (super fast!)

---

### Focused testing grid:
```json
{
  "0dte_params": {
    "breakeven_trigger_percent": [7, 10],
    "trail_method": ["pullback_percent"],
    "pullback_percent": [15, 20, 25],
    "native_trail_percent": [20],
    "psar_enabled": [false],
    "rsi_hook_enabled": [false]
  },
  "regular_params": {
    "breakeven_trigger_percent": [10, 12],
    "trail_method": ["pullback_percent", "atr"],
    "pullback_percent": [10, 12],
    "atr_period": [14],
    "atr_multiplier": [1.5],
    "native_trail_percent": [25, 30],
    "psar_enabled": [false],
    "rsi_hook_enabled": [false]
  }
}
```
**Result:** 
- 0DTE: 2 × 3 × 1 = 6 tests
- Regular: 2 × 2 × 2 × 2 = 16 tests
- Total: 22 tests (~5-10 min)

---

## 🚨 REQUIRED PARAMETERS

Even in custom mode, you MUST include these parameters:

### For 0DTE and Regular:
```json
{
  "breakeven_trigger_percent": [...],
  "trail_method": [...],
  "pullback_percent": [...],
  "atr_period": [...],
  "atr_multiplier": [...],
  "native_trail_percent": [...],
  "psar_enabled": [...],
  "psar_start": [0.02],
  "psar_increment": [0.02],
  "psar_max": [0.2],
  "rsi_hook_enabled": [...],
  "rsi_period": [14],
  "rsi_overbought": [70],
  "rsi_oversold": [30]
}
```

**Pro tip:** If you're not testing PSAR/RSI, just set them to single values:
```json
"psar_enabled": [false],
"rsi_hook_enabled": [false]
```

---

## 📁 FILE LOCATIONS

After you move the unified optimizer to your repo:

```
Discord_To_IBKR_3/
├── backtester/
│   ├── parameter_optimizer_unified.py       ← Main file
│   ├── param_grid_unified.json              ← Custom params template
│   │
│   └── optimization_results/
│       └── TIMESTAMP/
│           ├── unified_results.csv
│           └── unified_optimization_summary.txt
```

---

## ⚡ SPEED COMPARISON

| Mode | 0DTE Tests | Regular Tests | Time |
|------|-----------|---------------|------|
| **Quick** | 8 | 24 | ~8-10 min |
| **Full** | 100 | 240 | ~1-2 hours |
| **Custom (minimal)** | 1 | 1 | ~30 sec |
| **Custom (focused)** | 6 | 16 | ~5-10 min |

---

## 🎪 SUMMARY

✅ **YES, there is a quick test mode** - just use `--quick` flag

✅ **YES, there is a custom params file** - `param_grid_unified.json`

✅ **YES, you can use legacy files** - but unified is better for mixed signals

**Recommended workflow:**
1. Quick test all traders: `--quick` (10 min each)
2. Full test the good ones: no flags (1-2 hours)
3. Fine-tune winners: `--params custom.json` (focused testing)

**For your situation:**
```bash
# Start here - test all 11 traders quickly
for trader in goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf; do
    python backtester/parameter_optimizer_unified.py \
        --quick \
        --signals "backtester/channel_signals/${trader}_signals.txt"
done

# Then full test the 2-3 profitable ones
python backtester/parameter_optimizer_unified.py \
    --signals "backtester/channel_signals/expo_signals.txt"  # If Expo was good
```
