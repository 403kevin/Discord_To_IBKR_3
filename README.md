Project Revision 3: Discord to IBKR Bot
This document serves as the internal "Source of Truth" for the project. Its purpose is to keep development focused and to provide essential context for any AI assisting with the code.

1. Core Architecture Overview
This bot is a custom-built, Python-based automated options trading system. Its primary function is to connect a user account on Discord to an Interactive Brokers (IBKR) account to execute trades based on signals from specific channels.

CRITICAL ARCHITECTURAL NOTE: The bot's method for scraping Discord is a custom polling engine located in discord_interface.py. It operates by making rapid, sequential HTTP GET requests to Discord's API to fetch recent messages.

It IS: A discreet, lightweight polling system designed to "fly under the radar."

It IS NOT: An event-driven bot that maintains a persistent websocket connection using a library like discord.py.

All future development must respect this core architectural choice.

2. Current Accomplishments (As of 2025-09-01)
The following components of the system are considered functional and stable in the current codebase:

IBKR Connection: Successfully connects to TWS/Gateway and handles communication.

Trade Entry: A proven, reliable two-step entry logic is in place:

Place a MarketOrder to enter the position.

Immediately place a separate TrailOrder as the safety net.

Discord Polling: The custom polling engine successfully scrapes new messages from target channels.

Position Monitoring: A basic monitoring loop is functional, which includes a dynamic breakeven stop adjustment.

Telegram Notifications: The bot can successfully send trade confirmation messages.

Configuration: All settings are centrally managed within the config.py class.

3. Master To-Do List (Priority Order)
All development will follow this prioritized roadmap.

Priority 1: Refactor for Stability (HIGH)
Goal: Break down the monolithic main.py into a modular, single-responsibility structure.

Action: Create a new file structure with bot_engine/, interfaces/, and services/ directories to isolate components and make future upgrades safer and simpler.

Priority 2: Integrate FinBERT Sentiment Filter (HIGH)
Goal: Add a sentiment analysis step to the trade decision process.

Action: Integrate the sentiment_analyzer.py module. Before executing a trade, the bot will fetch news, analyze sentiment, and veto the trade if the score is unfavorable. The score will be added to the Telegram notification.

Priority 3: Remove Internal Hard Stop (MEDIUM)
Goal: Eliminate the risk of a race condition between the bot's internal stop and the broker's native trail order.

Action: Surgically remove the hard_stop_loss_percent logic from the position_monitor component.

Priority 4: Build Backtesting Engine (FUTURE)
Goal: Create a robust system to test the trading strategy against historical data.

Action: Develop a new, separate backtesting tool, likely using a professional data source like Databento. This is a future task to be addressed after the live bot is stable.
