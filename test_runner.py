# test_runner.py
import logging
import time
import sys

try:
    from services.config import Config
    from services.sentiment_analyzer import SentimentAnalyzer
    from interfaces.telegram_notifier import TelegramNotifier
    from interfaces.ib_interface import IBInterface
    from interfaces.discord_interface import DiscordInterface
    print("✅ [SUCCESS] All modules imported successfully.")
except ImportError as e:
    print(f"❌ [FATAL] Failed to import a required module: {e}")
    sys.exit(1)

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def partially_redact(token):
    """Helper function to show the start and end of a token for debugging."""
    if not token or not isinstance(token, str) or len(token) < 8:
        return "Invalid or too short to display"
    return f"{token[:4]}...{token[-4:]}"

class TestRunner:
    """
    A smarter test runner with enhanced diagnostics to verify the values
    being loaded from the .env file.
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
        """
        NEW: Now provides diagnostic output for the loaded tokens.
        """
        try:
            self.config = Config()
            
            # Diagnostic prints
            print(f"   [DIAGNOSTIC] Discord Token Loaded: {partially_redact(self.config.discord_user_token)}")
            print(f"   [DIAGNOSTIC] Telegram Bot Token Loaded: {partially_redact(self.config.telegram_bot_token)}")
            print(f"   [DIAGNOSTIC] Telegram Chat ID Loaded: {partially_redact(self.config.telegram_chat_id)}")

            if not all([self.config.discord_user_token, self.config.telegram_bot_token, self.config.telegram_chat_id]):
                return False, "Config class loaded, but one or more critical tokens/IDs are empty or missing. Check your .env file formatting."
            
            return True, "Successfully loaded .env file and initialized Config class."
        except Exception as e:
            return False, f"Failed to load configuration. Error: {e}. Ensure your .env file exists and is correctly formatted."

    def test_02_sentiment_analyzer(self):
        if not self.config: return False, "Skipped. Configuration failed to load."
        print("Initializing VADER sentiment analyzer... (This may require a one-time download)")
        try:
            analyzer = SentimentAnalyzer()
            if analyzer.analyzer:
                return True, "VADER sentiment analyzer initialized successfully."
            else:
                return False, "SentimentAnalyzer initialized, but the core analyzer is missing."
        except Exception as e:
            return False, f"Failed to initialize VADER. Error: {e}. Check your internet connection for the initial NLTK download."

    def test_03_telegram_notifier(self):
        if not self.config: return False, "Skipped. Configuration failed to load."
        try:
            notifier = TelegramNotifier(self.config)
            test_message = f"✅ Test message from the bot at {time.strftime('%Y-%m-%d %H:%M:%S')}. If you see this, the Telegram notifier is working."
            success = notifier.send_message(test_message)
            if success:
                return True, "Successfully sent a test message to your Telegram chat."
            else:
                return False, "Failed to send Telegram message. The API rejected the request. Please verify the tokens displayed in the diagnostic check above."
        except Exception as e:
            return False, f"An error occurred while testing Telegram. Error: {e}"

    def test_04_ibkr_connection(self):
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
        if not self.config: return False, "Skipped. Configuration failed to load."
        try:
            if not self.config.profiles:
                return False, "No profiles found in config. Cannot test Discord connection."
            channel_id_to_test = self.config.profiles[0]['channel_id']
            print(f"Testing Discord token by fetching info for channel {channel_id_to_test}...")
            discord_interface = DiscordInterface(self.config, callback=None)
            success = discord_interface.check_connection()
            if success:
                return True, "Discord token appears to be valid. Connection successful."
            else:
                return False, "Failed to connect to Discord API. Check the DISCORD_AUTH_TOKEN displayed in the diagnostic check above."
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

