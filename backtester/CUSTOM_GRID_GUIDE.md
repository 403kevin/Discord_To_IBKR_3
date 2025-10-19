# 📋 CUSTOM PARAMETER GRID TEMPLATES - GUIDE

## 🎯 Three Templates for Three Use Cases

You now have **THREE** custom parameter grid templates:

1. **`param_grid_regular.json`** - For regular options (TSLA, NVDA, AAPL, etc.)
2. **`param_grid_0DTE.json`** - For 0DTE options (SPX/SPY same-day expiry)
3. **`param_grid_template.json`** - OLD file from repo (DO NOT USE - outdated)

---

## 🚀 QUICK START

### **For Regular Options (held hours to days):**
```bash
# Edit the template
nano backtester/param_grid_regular.json

# Run with custom parameters
python backtester/parameter_optimizer.py --params param_grid_regular.json
```

### **For 0DTE Options (held 15-60 minutes):**
```bash
# Edit the template
nano backtester/param_grid_0DTE.json

# Run with custom parameters
python backtester/parameter_optimizer_0DTE.py --params param_grid_0DTE.json
```

---

## 📊 PARAMETER COMPARISON

| Parameter | Regular Options | 0DTE Options | Why Different? |
|-----------|----------------|--------------|----------------|
| **Breakeven Trigger** | 10-15% | 3-7% | 0DTE moves fast, needs tighter trigger |
| **Pullback %** | 8-12% | 5-10% | 0DTE can't afford wide pullbacks |
| **Native Trail** | 25-35% | 15-25% | 0DTE needs tighter safety net |
| **PSAR** | Test both | Disabled | Not useful for ultra-short holds |
| **RSI** | Test both | Disabled | Not useful for ultra-short holds |
| **ATR Trail** | Test both | Disabled | Too slow for 0DTE |

---

## 🔧 HOW TO CUSTOMIZE

### **Example 1: Test Specific Hunch (Regular Options)**

You think **10% breakeven + 10% pullback** works best:

```json
{
  "breakeven_trigger_percent": [10],
  "trail_method": ["pullback_percent"],
  "pullback_percent": [10],
  "native_trail_percent": [25],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

**Result:** Tests only **1 combination** (super fast!)

---

### **Example 2: Narrow Down After Quick Test**

Quick test showed 10-15% breakeven works, now you want to fine-tune:

```json
{
  "breakeven_trigger_percent": [8, 10, 12, 15],
  "trail_method": ["pullback_percent"],
  "pullback_percent": [8, 10, 12],
  "native_trail_percent": [25, 30],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

**Result:** Tests **24 combinations** (4 × 3 × 2)

---

### **Example 3: Compare Trail Methods**

Test pullback vs ATR:

```json
{
  "breakeven_trigger_percent": [10],
  "trail_method": ["pullback_percent", "atr"],
  "pullback_percent": [10],
  "atr_period": [14],
  "atr_multiplier": [1.5, 2.0],
  "native_trail_percent": [25],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

**Result:** Tests **4 combinations** (1 × 2 × 2)

---

### **Example 4: 0DTE Fine-Tuning**

After testing, you found 5% breakeven + 7% pullback works:

```json
{
  "breakeven_trigger_percent": [4, 5, 6],
  "trail_method": ["pullback_percent"],
  "pullback_percent": [6, 7, 8],
  "native_trail_percent": [18, 20, 22],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

**Result:** Tests **27 combinations** (3 × 3 × 3)

---

## 🎮 COMPLETE WORKFLOW EXAMPLE

### **Phase 1: Quick Discovery (10 minutes)**
```bash
# Test with built-in quick mode
python backtester/parameter_optimizer.py --quick

# Review results
cat backtester/optimization_results/TIMESTAMP/optimization_summary.txt
```

**Finding:** Breakeven 10-15% and pullback 8-12% look promising.

---

### **Phase 2: Custom Refinement (30 minutes)**

Edit `param_grid_regular.json`:
```json
{
  "breakeven_trigger_percent": [8, 10, 12, 15],
  "trail_method": ["pullback_percent"],
  "pullback_percent": [8, 10, 12],
  "native_trail_percent": [25, 30, 35],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

Run:
```bash
python backtester/parameter_optimizer.py --params param_grid_regular.json
```

**Tests:** 4 × 3 × 3 = **36 combinations**

---

### **Phase 3: Final Optimization (15 minutes)**

Results show 10% breakeven + 10% pullback + 30% native trail is best.

Test variations around that:
```json
{
  "breakeven_trigger_percent": [9, 10, 11],
  "trail_method": ["pullback_percent"],
  "pullback_percent": [9, 10, 11],
  "native_trail_percent": [28, 30, 32],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

**Tests:** 3 × 3 × 3 = **27 combinations**

---

### **Phase 4: Apply to Production**

Use the winning configuration in `services/config.py`:
```python
"exit_strategy": {
    "breakeven_trigger_percent": 0.10,
    "trail_method": "pullback_percent",
    "native_trail_percent": 0.30,
    "trail_settings": {
        "pullback_percent": 0.10
    }
}
```

---

## 💡 PRO TIPS

### **1. Start Broad, Then Narrow**
- Quick mode (16 tests) → Identify general patterns
- Custom grid (20-50 tests) → Refine promising areas
- Fine-tune (10-30 tests) → Optimize exact values

### **2. Test One Variable at a Time**
Bad:
```json
{
  "breakeven_trigger_percent": [5, 10, 15, 20],
  "pullback_percent": [5, 10, 15, 20],
  "native_trail_percent": [20, 25, 30, 35]
}
```
**Tests:** 4 × 4 × 4 = 64 combinations (hard to interpret)

Good:
```json
{
  "breakeven_trigger_percent": [10],
  "pullback_percent": [5, 10, 15, 20],  // Focus on this
  "native_trail_percent": [25]
}
```
**Tests:** 4 combinations (easy to see what matters)

### **3. Keep Native Trail Fixed**
The native trail is your safety net. Don't test too many values:
```json
{
  "native_trail_percent": [25]  // Just pick one and forget it
}
```

### **4. PSAR/RSI Usually Don't Help**
Unless you have a specific reason, disable them:
```json
{
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```

---

## 🚨 COMMON MISTAKES

### **❌ Mistake #1: Too Many Parameters**
```json
{
  "breakeven_trigger_percent": [5, 10, 15, 20],
  "pullback_percent": [8, 10, 12, 15],
  "native_trail_percent": [20, 25, 30, 35],
  "psar_enabled": [true, false],
  "rsi_hook_enabled": [true, false]
}
```
**Tests:** 4 × 4 × 4 × 2 × 2 = **512 combinations** (takes hours!)

### **✅ Better:**
```json
{
  "breakeven_trigger_percent": [10, 15],
  "pullback_percent": [10, 12],
  "native_trail_percent": [25],
  "psar_enabled": [false],
  "rsi_hook_enabled": [false]
}
```
**Tests:** 2 × 2 = **4 combinations** (takes minutes!)

---

### **❌ Mistake #2: Using 0DTE Parameters for Regular Options**
```json
{
  "breakeven_trigger_percent": [3, 5],  // ❌ Too tight for regular options!
  "pullback_percent": [5, 7]            // ❌ Too tight!
}
```

### **✅ Correct:**
Use `param_grid_regular.json` for regular options:
```json
{
  "breakeven_trigger_percent": [10, 15],  // ✅ Appropriate for multi-hour holds
  "pullback_percent": [10, 12]            // ✅ Appropriate
}
```

---

### **❌ Mistake #3: Testing ATR for 0DTE**
```json
{
  "trail_method": ["pullback_percent", "atr"]  // ❌ ATR doesn't work for 0DTE
}
```

### **✅ Correct:**
```json
{
  "trail_method": ["pullback_percent"]  // ✅ Only pullback for 0DTE
}
```

---

## 📁 FILES SUMMARY

| File | Purpose | Use With |
|------|---------|----------|
| **param_grid_regular.json** | Regular options template | `parameter_optimizer.py` |
| **param_grid_0DTE.json** | 0DTE options template | `parameter_optimizer_0DTE.py` |
| **param_grid_template.json** | OLD repo file | ❌ DELETE or ignore |

---

## 🎯 DECISION TREE

```
Are you testing 0DTE (SPX/SPY same-day)?
├─ YES → Edit param_grid_0DTE.json
│         Run: python backtester/parameter_optimizer_0DTE.py --params param_grid_0DTE.json
│
└─ NO → Edit param_grid_regular.json
        Run: python backtester/parameter_optimizer.py --params param_grid_regular.json
```

---

## ✅ NEXT STEPS

1. **Delete or ignore** `param_grid_template.json` (outdated)
2. **Use** `param_grid_regular.json` for most options
3. **Use** `param_grid_0DTE.json` only for SPX/SPY 0DTE
4. **Start with quick mode** before custom grids
5. **Keep custom grids small** (under 50 combinations)

---

**Remember:** The goal is to find the best parameters FASTER, not test everything exhaustively!
