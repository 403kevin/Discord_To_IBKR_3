Automated Options Trading Bot Suite
An advanced, Python-based system for executing options trades on Interactive Brokers (IBKR). This repository contains the framework for two distinct but related trading bots: a real-time Discord Signal Scraper and a sophisticated, multi-layered Hybrid Strategy Bot.

🤖 Bot 1: The Discord Signal Scraper
This bot is designed to parse and execute trading signals from designated Discord channels in real-time. It is built to be a stable, reliable foundation for following signal providers where official bot invitations are not possible.

Key Features
Efficient Sequential Polling: Utilizes a single user account to efficiently scrape multiple channels in a rapid, sequential loop, designed to fly under the radar.

Per-Channel Strategy Assignment: Assign unique, custom exit strategies to each Discord channel via a flexible config.py, allowing for tailored risk management for each signal provider.

Modular Exit Logic: Supports multiple exit strategies out-of-the-box, including:

Bracket Orders: Automatically set take-profit and stop-loss orders.

Multi-Stage Dynamic Trail: An advanced trail with breakeven triggers and adaptive stops.

"Modify-on-Condition" Native Trails: Places a wide, server-side trail on entry and intelligently tightens it once profit or sentiment targets are met.

End-of-Day Failsafe: Automatically closes all open positions at a configurable time to manage overnight risk.

🧠 Bot 2: The Hybrid Strategy Bot
This is a more advanced system designed to execute trades based on a sophisticated, multi-layered strategy. It acts on a consensus of technical indicators, filtered by real-time market internals and sentiment analysis.

Strategy Overview: The Hybrid Model
This bot's core strategy is a hybrid model that combines high-level technical analysis with low-latency market internals to identify and execute high-probability intraday trades. The process is a two-layer filter:

The Strategic Filter (The "Setup"): First, the bot uses a consensus model of multiple TradingView indicators to identify a favorable strategic bias (bullish or bearish) for a given asset. This tells the bot what to look for.

The Tactical Trigger (The "Entry"): Once a strategic bias is established, the bot switches to a low-latency execution mode. It then waits for a tactical trigger based on real-time market internals (like the NYSE $TICK and the $VIX) and immediate price action, as used by professional scalpers. This tells the bot precisely when to enter.

Signal Sources
TradingView Indicators: A consensus model based on a wide array of indicators sent via webhooks:

SuperTrend AI, Lorentzian Classification, AlphaTrend, Smart Money Concepts, Squeeze Momentum, Options Series, SPX Intraday Mood, and IV Rank/Percentile.

Alternative Data Events: Architecture will support signals from SEC filings (Form 4, 8-K), Unusual Options Activity (UOA), and real-time news headlines.

AI-Powered Sentiment Analysis: As a final sanity check, the bot will use a FinBERT model to analyze real-time news and social media sentiment (including niche sources like Donald Trump’s Truth Social posts), vetoing any trade that goes against strong, immediate market chatter.

🛠️ Shared Components & Technology
Both bots are built upon a shared set of powerful, modular components:

Execution Core (ib_interface.py): A robust interface for all order submission and position management with Interactive Brokers.

Risk Management (trailing_stop_manager.py): The advanced, multi-stage trailing stop logic is designed to be portable and usable by both bots.

Infrastructure & Logging: Both bots will utilize a shared system for Telegram alerts and detailed CSV trade logging for performance analysis, including trader names and strategy details.

Backtesting Engine: A dedicated engine will be built to simulate and validate strategies for both bots against historical IBKR data.

🚀 General Setup & Installation
Clone the Repository.

Create a Virtual Environment (python -m venv venv).

Install Dependencies (pip install -r requirements.txt).

Configure Secrets: Create a .env file for your private Discord user token, IBKR account details, and Telegram keys.

Configure Strategies: Adjust the config.py file to define your channel profiles, indicator settings, and strategy rules.

Running the Bot
The bot is initiated by running the main script, which will launch the appropriate listeners.

python main.py
