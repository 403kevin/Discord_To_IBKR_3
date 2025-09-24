



Project Revision 3: Discord to IBKR Bot (Internal "Source of Truth")
This document serves as the binding directive for all development on this project. Its purpose is to provide essential context, define the project's philosophy, and prevent architectural drift. All AI assisting with this code must adhere strictly to the principles laid out herein.

1. Guiding Principles & AI Commandments
The Commandment
The core logic and philosophy of this project MUST be preserved. The AI is a surgical tool, not an architect. Only specific, commanded tasks may be executed. The AI's behavior is governed by the "Surgical Precision" prompt template, which may be modified within this section.

What Has Failed (A History of AI Errors)
A review of our development history has revealed several critical AI failures. These patterns must not be repeated:

Architectural Misunderstanding: The AI incorrectly assumed a standard, event-driven discord.py architecture, ignoring the project's actual custom polling engine.

Configuration Hallucination: The AI invented a non-existent config.json file, failing to recognize that all settings are managed within the config.py class.

Destructive Refactoring: The AI attempted to completely rebuild working files (main.py, ib_interface.py) instead of performing the requested surgical modifications.

What Not To Do (Binding AI Constraints)
DO NOT replace or suggest replacing the custom polling engine in discord_interface.py.

DO NOT deviate from the established settings structure in the config.py class.

DO NOT perform large-scale refactoring unless it is the explicitly stated Primary Task.

DO NOT assume a standard library is being used when the architecture specifies a custom implementation.

2. Core Architecture Overview
This bot is a custom-built, Python-based automated options trading system. Its primary function is to connect a user account on Discord to an Interactive Brokers (IBKR) account to execute trades.

CRITICAL NOTE: The method for scraping Discord is a custom polling engine. It operates by making rapid, sequential HTTP GET requests to Discord's API to fetch recent messages. It is designed to be discreet and lightweight.

3. Current Accomplishments (As of 2025-09-01)
The following components of the system are considered functional and stable in the current codebase:

IBKR Connection: Successfully connects to TWS/Gateway and handles communication.

Trade Entry: A proven, reliable two-step entry logic is in place:

Place a MarketOrder to enter the position.

Immediately place a separate TrailOrder as the safety net.

Discord Polling: The custom polling engine successfully scrapes new messages from target channels.

Position Monitoring: A basic monitoring loop is functional, which includes a dynamic breakeven stop adjustment.

Telegram Notifications: The bot can successfully send trade confirmation messages.

Configuration: All settings are centrally managed within the config.py class.

4. Project Philosophy & Signal Parsing
Philosophy
The bot is designed to be flexible and interpret a wide variety of trading signal formats.

Buzzwords: Core actions like buy, trim, and sell are defined as buzzwords in config.py, allowing for easy customization.

Signal Format Legend
Signals can appear in numerous formats. The parser must be able to handle any variation of the following components, whether they are on the same line, separate lines, or embedded in other text.

A: Action Buzzword (buy, trim, sell, etc.)

B: Ticker (SPX, AAPL, etc.)

C: Strike & Right (5000C, 150P, etc.)

D: Expiry (12/25, 0DTE, mm/dd, etc.)

Examples of Valid Formats:

ABCD (e.g., BTO SPX 5000C 12/25)

ABC (e.g., Buy AAPL 150P)

A B C D (components on separate lines)

ABC D (expiry on a new line)

Ambiguous Expiry Toggle: A toggle in config.py will control behavior for signals without an expiry (like ABC). If True, the bot will execute the trade using the next available weekly expiration date for that ticker.

5. Master To-Do List (Priority Order)
High Priority
Refactor for Stability: Break down the monolithic main.py into a modular, single-responsibility structure (bot_engine/, interfaces/, services/). This is the immediate next step.

Internal Close-All Logic: Implement a safety mechanism that periodically polls the IBKR portfolio. If an "oversold" condition is detected (e.g., negative position on a contract), the bot must liquidate all positions to return the portfolio to a flat state.

Integrate FinBERT Sentiment Filter: Add sentiment analysis to the trade decision process. Veto trades with unfavorable scores and add the score to the Telegram notification.

Medium Priority
Remove Internal Hard Stop: Surgically remove the hard_stop_loss_percent logic to prevent race conditions with the native trail order.

Refine Telegram Signals: Modify the notifier to only send a success message after receiving a confirmed fill event from the IBKR API, not just on order placement.

Internal Kill Switch: Add a feature in config.py to define a max number of consecutive losses from a single Discord channel. If the limit is reached, the bot will stop taking signals from that channel for a configurable cooldown period.

Master Shutdown Command: Implement a listener in a private Discord channel. If a "terminate" command is detected, the script will gracefully shut down all processes.

Future Development
Build Backtesting Engine: Develop a robust backtester using Databento or IBKR historical data. The first step will be a small test script to prove data can be fetched for expired options.

AI Backtest Analysis: Create a separate AI prompt/script that can analyze the backtest results and generate recommended settings for each trader/channel.

Alternative Sentiment Filters: Integrate additional sentiment sources (e.g., Reddit's r/wallstreetbets sentiment, StockTwits) to work alongside FinBERT.

Alternative to Pullback: Explore integrating TradingView indicators (e.g., ATR, SuperTrend) via webhooks or an API as an alternative risk management strategy to a fixed pullback percentage.

6. External Resources
Interactive Brokers Order Types: Official documentation defining the behavior of all supported order types.





Final Core File Structure
main.py - The Boss. Its only job is to start the bot and manage the high-level application flow. It will be very short and simple.

README.md - The Instruction Manual. Our complete, up-to-date project guide.

requirements.txt - The Shopping List. All project dependencies.

.gitignore - The Secret Keeper. Protects your sensitive files.

bot_engine/ - The Specialist Team. This new folder will contain the core logic modules.

signal_processor.py - The Decision Maker. Takes a raw signal and decides if it's a valid trade.

position_monitor.py - The Watchtower. Monitors all live positions for stop-loss adjustments.

trade_executor.py - The Trader. Takes a "GO" decision and executes the trade via the IB interface.

interfaces/ - The Communications Hub. This new folder will contain all modules that talk to the outside world.

discord_interface.py - The Ears. Polls Discord for new messages.

ib_interface.py - The Hands. Communicates with Interactive Brokers.

telegram_notifier.py - The Mouth. Sends you notifications.

services/ - The Toolbox. This new folder will contain all the shared utilities and helper modules.

config.py - The Settings. Loads and provides all configuration.

sentiment_analyzer.py - The Vibe Checker. Analyzes news sentiment.

custom_logger.py & trade_logger.py - The Diaries. Handle all logging.

message_parsers.py - The Translator. Decodes raw Discord messages.

trailing_stop_manager.py - The Safety Net. Manages stop-loss calculations.

utils.py - The Gadgets. Contains various helper functions.
