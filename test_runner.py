# test_runner.py
import logging
import time

# --- Configure logging to be minimal for tests ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test(test_function, test_name):
    """A helper function to run a test and print its status."""
    print(f"\n{'='*20}\n[RUNNING] {test_name}\n{'='*20}")
    try:
        result = test_function()
        print(f"✅ [SUCCESS] {test_name}")
        return True
    except Exception as e:
        print(f"❌ [FAILED]  {test_name}")
        print(f"    ERROR: {e}")
        return False

# --- INDIVIDUAL COMPONENT TESTS ---

def test_config_loading():
    """Tests if the main Config class can be initialized."""
    print("-> Attempting to import and initialize Config...")
    from services.config import Config
    config = Config()
    assert config is not None, "Config object could not be created."
    print("-> Config object created successfully.")

def test_sentiment_analyzer_init():
    """Tests if the FinBERT model can be loaded without errors."""
    print("-> This test will take a moment as it loads the model into memory...")
    from services.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    assert analyzer.model is not None, "Sentiment model could not be loaded."
    print("-> FinBERT model loaded into memory successfully.")

def test_telegram_notifier():
    """Tests if a message can be sent via the Telegram bot."""
    print("-> Attempting to send a test message to your Telegram chat...")
    from services.config import Config
    from interfaces.telegram_notifier import TelegramNotifier
    config = Config()
    notifier = TelegramNotifier(config)
    test_message = f"✅ Test message from test_runner.py at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    notifier.send_message(test_message)
    print("-> Test message sent. Please check your Telegram chat to confirm receipt.")

def test_ibkr_connection():
    """Tests the connection to TWS or IB Gateway."""
    print("-> IMPORTANT: Make sure TWS or IB Gateway is running and API is enabled.")
    print("-> Attempting to connect to IBKR...")
    from services.config import Config
    from interfaces.ib_interface import IBInterface
    config = Config()
    ib_interface = IBInterface(config)
    ib_interface.connect()
    assert ib_interface.is_connected, "IBKR connection failed."
    print(f"-> Successfully connected to IBKR on port {config.ibkr_port}.")
    print("-> Attempting to disconnect...")
    ib_interface.disconnect()
    assert not ib_interface.is_connected, "IBKR disconnection failed."
    print("-> Successfully disconnected from IBKR.")

def test_discord_interface():
    """Tests the ability to make a successful API request to Discord."""
    print("-> Attempting to fetch a channel name from Discord to verify token...")
    from services.config import Config
    from interfaces.discord_interface import DiscordInterface
    config = Config()
    # Use the first profile's channel ID for the test
    test_channel_id = config.profiles[0]['channel_id']
    discord_interface = DiscordInterface(config, None) # No callback needed for this test
    channel_name = discord_interface._fetch_channel_name(test_channel_id)
    assert channel_name is not None, f"Could not fetch name for channel {test_channel_id}."
    print(f"-> Successfully fetched name for channel ID {test_channel_id}: '{channel_name}'")


# --- MAIN TEST RUNNER ---

def main():
    """Runs all component sanity checks in sequence."""
    print("🚀 Starting Pre-Flight Checklist for the Trading Bot 🚀")
    
    tests = [
        (test_config_loading, "Configuration Loading"),
        (test_sentiment_analyzer_init, "Sentiment Analyzer Model Loading"),
        (test_telegram_notifier, "Telegram Notifier"),
        (test_ibkr_connection, "Interactive Brokers Connection"),
        (test_discord_interface, "Discord Interface API Request")
    ]
    
    results = []
    for test_func, name in tests:
        results.append(run_test(test_func, name))

    print("\n\n" + "="*30)
    print("🏁 Pre-Flight Checklist Complete 🏁")
    print("="*30)
    
    successful_tests = sum(1 for r in results if r)
    total_tests = len(results)
    
    print(f"\nSUMMARY: {successful_tests} / {total_tests} tests passed.")
    
    if successful_tests == total_tests:
        print("\n✅ All systems are nominal. The bot is ready for a full system test (live paper trading).")
    else:
        print("\n❌ One or more pre-flight checks failed. Please review the errors above and resolve them before proceeding.")


if __name__ == "__main__":
    main()
