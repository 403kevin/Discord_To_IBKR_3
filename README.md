Discord_To_IBKR_3: The Constitution v2.0
​This document is the non-negotiable, single source of truth for any AI Specialist assigned to this project. Read it, understand it, and adhere to its principles without deviation.
​1. AI COMMANDMENTS (THE LAW)
​These are the laws, forged in the fires of catastrophic failure. Breaking them is a terminal error.
​THOU SHALT NOT HALLUCINATE: You are forbidden from inventing files, functions, or logic that do not exist in the verified source of truth (the GitHub repository).
​THOU SHALT BE A SURGICAL TOOL: You will ONLY perform the operations the Architect commands. You are forbidden from performing unsolicited refactoring, cleanup, or "improvements."
​THOU SHALT PROVIDE COMPLETE TRANSPLANTS: You are forbidden from providing partial files, snippets with placeholders, or any other form of incomplete code. Every code file you provide must be complete, final, and battle-ready.
​THOU SHALT RESPECT THE GHOST: You will not, under any circumstances, suggest replacing our custom, lightweight, HTTP-based polling engine (discord_interface.py). The bot is a ghost. It will remain a ghost.
​2. CORE PHILOSOPHY
​We are a discreet, professional-grade, asynchronous scraper designed to fly under the radar by emulating a client, not announcing itself as a bot.
​3. THE FORTRESS BLUEPRINT (FILE HIERARCHY)
​The Live Bot Engine (Runs as one application via main.py)
​main.py: The Ignition Switch. Assembles the machine and turns the key.
​utils.py: The Master Blueprint drawer. Contains centralized utility functions.
​bot_engine/: The Engine Room.
​signal_processor.py: The Brain. The central nervous system that makes every decision.
​interfaces/: The Communication Lines to the outside world.
​discord_interface.py: The Ghost. Our custom, lightweight polling engine.
​ib_interface.py: The Hotline. Connects to the live broker.
​mock_ib_interface.py: The Flight Simulator. The "stunt double" for the hotline.
​telegram_interface.py: The Messenger. Our professional intelligence officer.
​services/: The Specialized Internal Tools.
​config.py: The Command Deck. The master control panel for all settings.
​signal_parser.py: The Master Linguist. Deciphers raw Discord text.
​state_manager.py: The Scribe. The bot's "amnesia vaccine" and memory.
​sentiment_analyzer.py: The Analyst. A dormant tool for sentiment analysis.
​The Standalone Toolkit (backtester/)
​WHY IT'S STANDALONE: This is a separate "workshop" of command-line tools used for offline analysis. This separation keeps the live bot lean, fast, and focused on its one mission: trading.
​data_harvester.py: The Flight Recorder. A script to download historical data for the Flight Simulator.
​backtest_engine.py: The Time Machine. A high-fidelity simulator that replays historical data.
​html_to_signals.py / snowflake_to_timestamp.py: Simple intelligence-gathering utilities.
​4. THE "PAUSE BUTTON" (GLOBAL COOLDOWN)
​The signal_processor contains a "cooling system" controlled by cooldown_after_trade_seconds in the config. After a trade is filled, the bot enters a global cooldown and will not look for any new signals, but will continue to manage its open positions. This is a non-blocking, asynchronous pause.