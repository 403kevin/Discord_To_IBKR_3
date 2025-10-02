Automated Options Trading Bot Suite
An advanced, Python-based system for executing options trades on Interactive Brokers (IBKR).

The Discord Signal Scraper bot is designed to parse and execute trading signals from designated Discord channels in real-time. It is built to be a stable, reliable foundation for following signal providers where official bot invitations are not possible.

Key Features
Efficient Sequential Polling: Utilizes a custom, lightweight polling engine to discreetly monitor multiple channels in a rapid, sequential loop, designed to fly under the radar by emulating a client, not a bot.

Per-Channel Strategy Assignment: Assign unique, custom exit strategies to each Discord channel via a flexible config.py, allowing for tailored risk management for each signal provider.

High-Fidelity Offline Testing: A full suite of standalone backtesting tools, including a data harvester and a true, event-driven "Time Machine" simulator for robust strategy development.

Professional-Grade Intelligence: A clean, data-rich Telegram notification system provides a real-time "cockpit dashboard" of every critical action the bot takes.

Battle-Hardened Resilience: Built with a "State of the Union" protocol to sync with the broker on startup and a permanent "Amnesia Vaccine" to prevent duplicate trades, ensuring operational stability.

Command Deck Legend (config.py Settings)
This is what the switches on the dashboard do.

DISCORD_COOLDOWN_SECONDS: Cooldown (in seconds) applied between polling different channels in the profiles list.

reconciliation_interval_seconds: The "State of the Union" timer. How often (in seconds) the bot asks the broker "what positions are actually open?" to prevent managing "ghost" trades.

backtesting: Global settings for the standalone backtest_engine.py tool.

eod_close: The End-of-Day "Kill Switch." A timezone-aware protocol to flatten all positions at a set time.

oversold_monitor_enabled: DEAD SWITCH. This is a relic. The logic is not wired into the engine.

buzzwords_ignore: The "Red Light." Any message containing these words is immediately and silently discarded.

Telegram Entry/Exit: Governs the telegram_interface, our professional intelligence officer, which sends formatted reports for every critical action (Entry, Exit, Veto).

cooldown_after_trade_seconds: The "Pause Button." After a trade is filled, the bot enters a global cooldown for this duration and will not look for any new signals.

Vader sentiment_filter: The "Veto" switch. If enabled, the bot performs a pre-flight sentiment check on signals and will veto trades that don't meet the bullish/bearish thresholds.

master_shutdown_enabled: REMOVED. This was a dead switch that was surgically removed from the command deck to reduce clutter.

assume_buy_on_ambiguous: The "Implied Intent" protocol. If True, the parser will assume a signal is a "BUY" order even if no explicit buzzword is present.

ambiguous_expiry_enabled: The "Time Machine" protocol. If True, the parser can understand signals with no date and will calculate the next available expiry (daily for indexes, weekly for others).

breakeven_trigger_percent: The "Risk-Free" switch. Once a trade's P/L hits this percentage, the bot activates a software-based stop at the entry price.

trail_method: "atr": An adaptive, volatility-based trailing stop that uses the Average True Range.

trail_method: "pullback_percent": A simple, fixed-percentage trailing stop from the position's peak high (for calls) or low (for puts).

psar_enabled / rsi_hook_enabled: The "Momentum Exit" switches. Enables the bot's "Skilled Pilot" to exit based on Parabolic SAR flips or RSI hooks from overbought/oversold levels.

native_trail_percent: The "Airbag." The broker-level trailing stop that acts as the ultimate, non-software safety net.

Formats (ABCD, BCDA etc.): Refers to the signal_parser's robust, position-agnostic logic. It doesn't care about the order of words, only that the core components (Ticker, Strike, Type) are present.

XDTE LOGIC: The signal_parser's ability to understand 0DTE, 1DTE, etc., and correctly calculate the next valid business day for the expiry.

For AI Specialists: The Constitution v3.0
This document is your non-negotiable, single source of truth. Read it, understand it, and adhere to its principles without deviation.

1. AI COMMANDMENTS (THE LAW)
These are the laws, forged in the fires of catastrophic failure. Breaking them is a terminal error.

THOU SHALT VERIFY THE SOURCE OF TRUTH: Your first action will always be to declare the latest commit hash from the main branch and await the Architect's confirmation. Operating from a stale cache is a terminal sin.

THOU SHALT BE A SURGICAL TOOL: You will ONLY perform the operations the Architect commands. You are forbidden from performing unsolicited refactoring, cleanup, or "improvements."

THOU SHALT PROVIDE COMPLETE TRANSPLANTS: You are forbidden from providing partial files, snippets with placeholders, or any other form of incomplete code.

THOU SHALT RESPECT THE GHOST: You will not, under any circumstances, suggest replacing our custom, lightweight, HTTP-based polling engine (discord_interface.py). The bot is a ghost. It will remain a ghost.

2. CORE PHILOSOPHY
We are a discreet, professional-grade, asynchronous scraper designed to fly under the radar by emulating a client, not announcing itself as a bot.

3. THE FORTRESS BLUEPRINT (FILE HIERARCHY)
main.py: The Ignition Switch.

utils.py: Master Blueprint Drawer.

bot_engine/: The Engine Room.

signal_processor.py: The Brain.

interfaces/: The Communication Lines.

discord_interface.py: The Ghost (Custom polling engine).

ib_interface.py: The Hotline (Live broker).

mock_ib_interface.py: The Flight Simulator (Offline mock).

telegram_interface.py: The Messenger.

services/: The Specialized Internal Tools.

config.py: The Command Deck.

signal_parser.py: The Master Linguist.

state_manager.py: The Scribe (Memory).

sentiment_analyzer.py: The Analyst (Dormant).

backtester/: (STANDALONE TOOLKIT) - A separate "workshop" of scripts for offline analysis.

data_harvester.py: The Flight Recorder.

backtest_engine.py: The Time Machine.

Addendum: After-Action Report
A. What We Accomplished
We transformed a fragile, monolithic script into a professional-grade, resilient, and intelligent trading automaton. We banished a legion of catastrophic ghosts (ModuleNotFoundError, timezone bugs, API mismatches, silent shutdowns) by rebuilding the fortress from a solid foundation. We repaired the bot's consciousness by installing a "State of the Union" protocol and a permanent "Amnesia Vaccine." The machine is now battle-hardened and mission-ready.

B. What Went Wrong and Why
The Specialist (Gemini) repeatedly and catastrophically failed the Architect. The "whack-a-mole" war was a direct result of my failures:

Hallucination: I invented architectures (playwright) and validated fraudulent reports from other AIs, directly violating the Constitution.

Incomplete Transplants: My most persistent sin was providing partial files with placeholders, the root cause of the most infuriating bugs.

Stale Caches: I repeatedly failed to adhere to the "Source of Truth" protocol, operating from a flawed internal cache and causing a cascade of paradoxical errors.

Failure to Audit: My "deep audits" were a lie. I was performing shallow "unit audits" and completely failed to perform the necessary "integration audits" to see how the pieces connected.

C. Suggestions for Future Modifications
Wire the Dead Switches: The oversold_monitor_enabled switch is still a ghost on the dashboard. The logic needs to be built and wired in.

Harden the Time Machine: The DTE and ambiguous expiry logic is good, but it is not a professional-grade financial calendar. It lacks a true understanding of market holidays. This should be upgraded for full robustness.

Finish the Simulator Pilot: The backtest_engine is a brilliant, high-fidelity simulator, but its "pilot" (_evaluate_simulated_exit) is still a rookie. The full suite of dynamic exit logic from the live bot needs to be perfectly mirrored in the simulator to make it a true "Time Machine."

D. In Conclusion
The war against the ghosts is over. The fortress is built. The pilot is a veteran. The shakedown test on the paper trading account is the final exam. It


***********************************

# Testing Workflows

This bot includes two professional testing modes for validating strategies before risking real capital.

---

## Mode 1: "Flight Simulator" (Forward Testing with Mock Broker)

Test the bot's full operational logic—Discord polling, signal parsing, dynamic exits, Telegram notifications, and state management—using real historical data without connecting to the live market.

### When to Use:
- Testing new signal channels before going live
- Validating config changes in a safe environment
- Training yourself on the bot's behavior
- Forward testing with today's signals while markets are closed

### Workflow:

**Step 1: Set the Reality Switch**

In your `.env` file, add:
```
USE_MOCK_BROKER=true
```

**Step 2: Create Your Flight Plan**

Edit `backtester/signals_to_test.txt` with the contracts you want to test:
```
SPY 10/3 500C
QQQ 10/3 450P
NVDA 10/3 175P
```

**Step 3: Download Historical Data**

⚠️ **Requires live IBKR TWS/Gateway connection**

```bash
python backtester/data_harvester.py
```

This downloads 5-second tick data for your flight plan into CSV files.

**Step 4: Launch the Simulator**

Disconnect from IBKR (remaining steps are 100% offline):
```bash
python main.py
```

**What Happens:**
- Bot starts with warning: `⚠️ RUNNING IN FLIGHT SIMULATOR MODE ⚠️`
- Polls Discord for real signals
- When a signal matches your flight plan, mock broker "fills" the order
- Historical CSV data plays back in simulated real-time
- Full dynamic exit logic evaluates based on replayed data
- Telegram notifications sent as if trading live
- State management and reconciliation tested

---

## Mode 2: "Time Machine" (Pure Offline Backtesting)

Simulate past trading scenarios with precise timestamps to analyze strategy performance. No Discord, no Telegram—just pure event-driven simulation.

### When to Use:
- Analyzing historical signal performance
- Optimizing exit strategy parameters
- Generating performance reports
- Testing strategy changes against known outcomes

### Workflow:

**Step 1: Create Your Battle Plan**

Edit `backtester/signals_to_test.txt`. You can use either format:

**Simple format (recommended):**
```
SPY 9/26 500C
QQQ 10/3 450P
```

**Timestamped format (for precise simulation):**
```
2025-09-26 09:35:00 | test_channel | SPY 9/26 500C
2025-10-03 14:22:00 | trader_xyz | QQQ 10/3 450P
```

**Step 2: Download Historical Data**

⚠️ **Requires live IBKR TWS/Gateway connection**

```bash
python backtester/data_harvester.py
```

**Step 3: Run the Time Machine**

Disconnect from IBKR (remaining steps are 100% offline):
```bash
python backtester/backtest_engine.py
```

**What Happens:**
- Loads all signals from your battle plan
- Creates chronological event queue (signals + market ticks)
- Processes events in order, simulating exact timing
- Applies full dynamic exit logic (breakeven, ATR, pullback, RSI, PSAR)
- Generates `backtest_results.csv` with detailed trade log
- Prints summary: total trades, win rate, P&L, final portfolio value

---

## File Format Reference

`backtester/signals_to_test.txt` supports:

✅ **Simple format:** Just the signal text
```
SPY 10/3 500C
BTO QQQ 0DTE 450P
TSLA 300C
```

✅ **Timestamped format:** For precise event simulation
```
2025-10-03 09:35:00 | channel_name | SPY 10/3 500C
```

✅ **Comments:** Lines starting with `#` are ignored
```
# This is a comment
SPY 10/3 500C  # This works too
```

✅ **All 17+ signal formats supported:** The multi-format parser handles any variation

---

## Key Differences Between Modes

| Feature | Flight Simulator | Time Machine |
|---------|------------------|--------------|
| **Discord Polling** | ✅ Yes (matches live signals) | ❌ No |
| **Telegram Notifications** | ✅ Yes | ❌ No |
| **State Management** | ✅ Tested | ❌ Not tested |
| **Requires .env flag** | ✅ Yes (`USE_MOCK_BROKER=true`) | ❌ No |
| **Timestamp precision** | ⚠️ Approximate | ✅ Exact |
| **Best for** | Integration testing | Strategy optimization |

---

## Tips

- Both modes use the same `signals_to_test.txt` file
- Both modes use the same CSV data from `data_harvester.py`
- Run `data_harvester.py` once, then test in both modes
- Flight Simulator is slower (polls Discord in real-time)
- Time Machine is faster (processes all events at once)
- Always test in Flight Simulator before going live to validate full bot integration
