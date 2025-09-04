# test_runner.py
import logging
import time
import os
import sys

# This is crucial to allow the script to find our new modules
# in the subdirectories.
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from services.config import Config
    from services.sentiment_analyzer import SentimentAnalyzer
    from interfaces.telegram_notifier import TelegramNotifier
    from interfaces.ib_interface import IBInterface
    from interfaces.discord_interface import DiscordInterface
    print("✅ [SUCCESS] All modules imported successfully.")
except ImportError as e:
    print(f"❌ [FATAL] Failed to import a required module: {e}")
    print("Please ensure you are running this script from the root project directory and all files are in place.")
    sys.exit(1)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestRunner:
    """
    A simple script to run pre-flight checks on all critical components
    of the trading bot to ensure they are configured and working correctly
    before launching the main application.
    """
    def __init__(self):
        self.config = None
        self.results = {}

    def run_all_tests(self):
        """Runs all checks in sequence and prints a final report."""
        print("\n--- Starting Pre-Flight System Checks ---")
        
        tests = [
            self.test_01_config_and_env,
            self.test_02_sentiment_analyzer,
            self.test_03_telegram_notifier,
            self.test_04_ibkr_connection,
            self.test_05_discord_connection
        ]

        for test in tests:
            try:
                test_name = test.__name__
                print(f"\n--- RUNNING: {test_name} ---")
                success, message = test()
                self.results[test_name] = {"success": success, "message": message}
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"--- RESULT: {status} ---")
                print(message)
                if not success and "FATAL" in test_name:
                    print("\nAborting due to fatal error.")
                    break
            except Exception as e:
                self.results[test.__name__] = {"success": False, "message": f"An unexpected exception occurred: {e}"}
                print(f"--- RESULT: ❌ FAILED ---")
                print(f"An unexpected exception occurred: {e}")

        self._print_final_report()

    def test_01_config_and_env(self):
        """Checks if the .env file can be read and the Config class can be initialized."""
        try:
            self.config = Config()
            if not all([self.config.discord_user_token, self.config.telegram_bot_token, self.config.telegram_chat_id]):
                return False, "Config class loaded, but one or more critical tokens/IDs are missing. Check your .env file."
            return True, "Successfully loaded .env file and initialized Config class."
        except Exception as e:
            return False, f"Failed to load configuration. Error: {e}. Ensure your .env file exists and is correctly formatted."

    def test_02_sentiment_analyzer(self):
        """Checks if the FinBERT model can be loaded."""
        if not self.config: return False, "Skipped. Configuration failed to load."
        print("Loading FinBERT model... (This may take a moment)")
        try:
            analyzer = SentimentAnalyzer()
            if analyzer.model and analyzer.tokenizer:
                return True, "FinBERT model and tokenizer loaded successfully."
            else:
                return False, "SentimentAnalyzer initialized, but model or tokenizer is missing."
        except Exception as e:
            return False, f"Failed to load FinBERT model. Error: {e}. Check your internet connection and transformers installation."

    def test_03_telegram_notifier(self):
        """Checks if a test message can be sent via Telegram."""
        if not self.config: return False, "Skipped. Configuration failed to load."
        try:
            notifier = TelegramNotifier(self.config)
            test_message = f"✅ Test message from the bot at {time.strftime('%Y-%m-%d %H:%M:%S')}. If you see this, the Telegram notifier is working."
            success = notifier.send_message(test_message)
            if success:
                return True, "Successfully sent a test message to your Telegram chat."
            else:
                return False, "Failed to send Telegram message. Check your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
        except Exception as e:
            return False, f"An error occurred while testing Telegram. Error: {e}"

    def test_04_ibkr_connection(self):
        """Checks if a connection can be established with TWS/Gateway."""
        if not self.config: return False, "Skipped. Configuration failed to load."
        print(f"Attempting to connect to IBKR at {self.config.ibkr_host}:{self.config.ibkr_port}...")
        ib_interface = IBInterface(self.config)
        try:
            ib_interface.connect()
            if ib_interface.is_connected:
                ib_interface.disconnect()
                return True, "Successfully connected to and disconnected from IBKR."
            else:
                return False, "Failed to connect to IBKR. Ensure TWS or Gateway is running and the API settings are correct."
        except Exception as e:
            return False, f"An error occurred while connecting to IBKR. Error: {e}. Is TWS/Gateway running?"

    def test_05_discord_connection(self):
        """Checks if the Discord user token is valid."""
        if not self.config: return False, "Skipped. Configuration failed to load."
        try:
            # We test one of the profile channel IDs
            channel_id_to_test = self.config.profiles[0]['channel_id']
            print(f"Testing Discord token by fetching info for channel {channel_id_to_test}...")
            discord_interface = DiscordInterface(self.config, callback=None)
            success = discord_interface.check_connection()
            if success:
                return True, "Discord token appears to be valid. Connection successful."
            else:
                return False, "Failed to connect to Discord API. Check your DISCORD_AUTH_TOKEN."
        except Exception as e:
            return False, f"An error occurred while testing Discord. Error: {e}"

    def _print_final_report(self):
        """Prints a summary of all test results."""
        print("\n\n--- PRE-FLIGHT CHECK FINAL REPORT ---")
        all_passed = True
        for name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"{status.ljust(8)} | {name}")
            if not result["success"]:
                all_passed = False
        
        print("-" * 37)
        if all_passed:
            print("✅ All systems nominal. The bot is ready for a full system test.")
        else:
            print("❌ One or more checks failed. Please review the errors above before proceeding.")
        print("-" * 37)

if __name__ == "__main__":
    runner = TestRunner()
    runner.run_all_tests()

