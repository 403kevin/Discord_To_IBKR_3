# 🎯 OPTIMIZER COMMAND CHEATSHEET

## Quick Reference: Which Command Do I Use?

---

## 📦 **THE THREE OPTIMIZERS**

### 1️⃣ Legacy 0DTE Optimizer
**File:** `parameter_optimizer_0DTE.py`
**Use for:** Pure SPX/SPY 0DTE signals only

### 2️⃣ Legacy Regular Optimizer  
**File:** `parameter_optimizer.py`
**Use for:** Pure stock option signals only

### 3️⃣ **Unified Super Optimizer** ⭐
**File:** `parameter_optimizer_unified.py`
**Use for:** Mixed signals OR testing multiple traders

---

## ⚡ COMMAND COMPARISON

### Quick Test Commands

| Optimizer | Command |
|-----------|---------|
| 0DTE (legacy) | `python backtester/parameter_optimizer_0DTE.py --quick` |
| Regular (legacy) | `python backtester/parameter_optimizer.py --quick` |
| **Unified** ⭐ | `python backtester/parameter_optimizer_unified.py --quick` |

### Full Test Commands

| Optimizer | Command |
|-----------|---------|
| 0DTE (legacy) | `python backtester/parameter_optimizer_0DTE.py` |
| Regular (legacy) | `python backtester/parameter_optimizer.py` |
| **Unified** ⭐ | `python backtester/parameter_optimizer_unified.py` |

### Custom Parameter Commands

| Optimizer | Command |
|-----------|---------|
| 0DTE (legacy) | `python backtester/parameter_optimizer_0DTE.py --params param_grid_0DTE.json` |
| Regular (legacy) | `python backtester/parameter_optimizer.py --params param_grid_regular.json` |
| **Unified** ⭐ | `python backtester/parameter_optimizer_unified.py --params param_grid_unified.json` |

---

## 🎯 YOUR SITUATION: Testing 11 Traders

### ❌ OLD WAY (Using Legacy Optimizers)

**Problem:** You don't know if each trader posts 0DTE, regular, or both

**Painful process:**
```bash
# Step 1: Manually check Goldman's signals
# Step 2: Realize he posts SPX → Use 0DTE optimizer
python backtester/parameter_optimizer_0DTE.py --quick --signals goldman_signals.txt

# Step 3: Manually check Expo's signals  
# Step 4: Realize he posts TSLA → Use regular optimizer
python backtester/parameter_optimizer.py --quick --signals expo_signals.txt

# Step 5: Manually check Zeus's signals
# Step 6: Realize he posts BOTH → ??? which optimizer ???
# Step 7: Run both optimizers separately
python backtester/parameter_optimizer_0DTE.py --quick --signals zeus_signals.txt
python backtester/parameter_optimizer.py --quick --signals zeus_signals.txt

# Repeat for all 11 traders... 😭
```

---

### ✅ NEW WAY (Using Unified Optimizer)

**Solution:** Let it auto-detect everything

**Simple process:**
```bash
# One command per trader, auto-handles everything
for trader in goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf; do
    python backtester/parameter_optimizer_unified.py \
        --quick \
        --signals "backtester/channel_signals/${trader}_signals.txt"
done
```

**What it does automatically:**
- Scans Goldman → Finds SPX → Tests 0DTE params
- Scans Expo → Finds TSLA → Tests regular params
- Scans Zeus → Finds BOTH → Tests BOTH params
- Generates unified report for each

---

## 📋 COMMON SCENARIOS

### Scenario 1: Goldman (Pure 0DTE Trader)

**Legacy way:**
```bash
python backtester/parameter_optimizer_0DTE.py --quick \
    --signals goldman_signals.txt
```

**Unified way:**
```bash
python backtester/parameter_optimizer_unified.py --quick \
    --signals goldman_signals.txt
```

**Result:** Same output, but unified gives you breakdown

---

### Scenario 2: Expo (Pure Stock Trader)

**Legacy way:**
```bash
python backtester/parameter_optimizer.py --quick \
    --signals expo_signals.txt
```

**Unified way:**
```bash
python backtester/parameter_optimizer_unified.py --quick \
    --signals expo_signals.txt
```

**Result:** Same tests, unified format is cleaner

---

### Scenario 3: Zeus (Mixed Trader)

**Legacy way:**
```bash
# Run both optimizers separately
python backtester/parameter_optimizer_0DTE.py --quick \
    --signals zeus_signals.txt

python backtester/parameter_optimizer.py --quick \
    --signals zeus_signals.txt

# Manually compare two separate reports 😫
```

**Unified way:**
```bash
python backtester/parameter_optimizer_unified.py --quick \
    --signals zeus_signals.txt
```

**Result:** ONE report showing both signal types

---

## 🚀 BATCH TESTING ALL TRADERS

### Legacy approach (PAINFUL):
```bash
# Have to manually categorize each trader first
# Then use different optimizers for each
# Then manually combine results
# 😭 Takes forever
```

### Unified approach (EASY):
```bash
#!/bin/bash
# test_all_traders.sh

traders=(goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf)

for trader in "${traders[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎯 Testing: $trader"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python backtester/parameter_optimizer_unified.py \
        --quick \
        --signals "backtester/channel_signals/${trader}_signals.txt"
    
    echo ""
done

echo "✅ All traders tested!"
echo "📊 Check backtester/optimization_results/ for reports"
```

**Run it:**
```bash
chmod +x test_all_traders.sh
./test_all_traders.sh
```

**Time:** ~10 min per trader = ~2 hours total for all 11

---

## 🎛️ MODE FLAGS

### Quick Mode: `--quick`
- **Purpose:** Fast preliminary testing
- **Time:** 8-10 minutes per trader
- **Tests:** 8-32 combinations (depends on signal mix)
- **Use when:** First time testing a trader

```bash
python backtester/parameter_optimizer_unified.py --quick
```

---

### Full Mode: (no flag)
- **Purpose:** Comprehensive optimization
- **Time:** 1-2 hours per trader
- **Tests:** 100-340 combinations (depends on signal mix)
- **Use when:** Found a profitable trader, want best params

```bash
python backtester/parameter_optimizer_unified.py
```

---

### Custom Mode: `--params <file>`
- **Purpose:** Test specific parameter ranges
- **Time:** Depends on your grid (can be 30 sec to 1 hour)
- **Tests:** Whatever you define
- **Use when:** Fine-tuning based on quick/full results

```bash
python backtester/parameter_optimizer_unified.py \
    --params custom_grid.json
```

---

### Specify Signals: `--signals <file>`
- **Purpose:** Test specific signal file (not default)
- **Combine with:** Any mode flag

```bash
python backtester/parameter_optimizer_unified.py \
    --quick \
    --signals backtester/channel_signals/expo_signals.txt
```

---

## 🔥 REAL-WORLD EXAMPLES

### Example 1: Quick test the default signals file
```bash
cd Discord_To_IBKR_3
python backtester/parameter_optimizer_unified.py --quick
```

---

### Example 2: Full test on Goldman after finding he's trash (to confirm)
```bash
python backtester/parameter_optimizer_unified.py \
    --signals backtester/channel_signals/goldman_signals.txt
```

---

### Example 3: Quick test Expo with custom focused parameters
Create `expo_focused.json`:
```json
{
  "regular_params": {
    "breakeven_trigger_percent": [10, 12],
    "trail_method": ["pullback_percent", "atr"],
    "pullback_percent": [10],
    "atr_period": [14],
    "atr_multiplier": [1.5, 2.0],
    "native_trail_percent": [30],
    "psar_enabled": [false],
    "rsi_hook_enabled": [false]
  }
}
```

Run:
```bash
python backtester/parameter_optimizer_unified.py \
    --params expo_focused.json \
    --signals backtester/channel_signals/expo_signals.txt
```

---

### Example 4: Night batch job testing all traders (full mode)
```bash
#!/bin/bash
# overnight_test.sh - Run full optimization on all 11 traders

for trader in goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf; do
    echo "Starting full test: $trader"
    
    python backtester/parameter_optimizer_unified.py \
        --signals "backtester/channel_signals/${trader}_signals.txt" \
        > "logs/${trader}_full_test.log" 2>&1
    
    echo "Completed: $trader"
done

echo "🎉 All 11 traders fully tested!"
```

**Run before bed:**
```bash
chmod +x overnight_test.sh
nohup ./overnight_test.sh &
```

**Wake up to:** 11 comprehensive optimization reports

---

## 📊 OUTPUT FILES

All three optimizers save results to: `backtester/optimization_results/TIMESTAMP/`

### Legacy Optimizers Output:
```
optimization_results/20251019_140530/
├── all_results.csv
└── optimization_summary.txt
```

### Unified Optimizer Output:
```
optimization_results/20251019_140530/
├── unified_results.csv                    ← All tests with signal_type column
└── unified_optimization_summary.txt       ← Breakdown by signal type
```

**Key difference:** Unified report shows:
- Overall stats
- 0DTE breakdown
- Regular breakdown  
- Top configs per type
- Cross-type comparison

---

## 🎪 DECISION FLOWCHART

```
Do you know if your trader posts 0DTE, regular, or both?
├─ YES: I know it's pure 0DTE
│   └─ Use legacy: parameter_optimizer_0DTE.py
│
├─ YES: I know it's pure regular options
│   └─ Use legacy: parameter_optimizer.py
│
├─ NO: Unknown or mixed
│   └─ Use unified: parameter_optimizer_unified.py ⭐
│
└─ Testing multiple traders with unknown types
    └─ Use unified: parameter_optimizer_unified.py ⭐
```

---

## ✅ BOTTOM LINE

### For your situation (11 traders, unknown signal types):

**Best command to start:**
```bash
for trader in goldman expo zeus arrow waxui qiqo nitro money-mo diesel prophet gandalf; do
    python backtester/parameter_optimizer_unified.py \
        --quick \
        --signals "backtester/channel_signals/${trader}_signals.txt"
done
```

**This will:**
- ✅ Test all 11 traders
- ✅ Auto-detect 0DTE vs regular for each
- ✅ Test each with optimal parameters
- ✅ Generate unified comparison reports
- ✅ Take ~2 hours total (10 min per trader)
- ✅ Show you which traders are worth following

**Then for the profitable traders:**
```bash
# Run full mode on the winners
python backtester/parameter_optimizer_unified.py \
    --signals backtester/channel_signals/<winner>_signals.txt
```

---

## 🚨 REMEMBER

- ✅ `--quick` flag = fast testing (8-10 min)
- ✅ No flag = full testing (1-2 hours)
- ✅ `--params` = custom parameters
- ✅ `--signals` = specify signal file
- ✅ Unified optimizer = auto-detects signal types
- ✅ Legacy optimizers = manual categorization required
