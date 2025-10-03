import logging
import asyncio
import os
import sys
from datetime import datetime
from bs4 import BeautifulSoup

# --- GPS FOR THE FORTRESS ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from services.config import Config
from services.signal_parser import SignalParser

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HTMLToSignals:
    """
    Converts exported Discord HTML files to the signals_to_test.txt format
    for backtesting. Useful for batch-processing historical Discord exports.
    """
    def __init__(self, config):
        self.config = config
        self.signal_parser = SignalParser(config)

    def parse_html_export(self, html_file_path):
        """
        Parses a Discord HTML export file and extracts messages with timestamps.
        
        Args:
            html_file_path: Path to the HTML file exported from Discord
            
        Returns:
            List of tuples: [(timestamp, message_text), ...]
        """
        messages = []
        
        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
            # Discord HTML exports typically have messages in divs with specific classes
            # This is a generic parser - adjust selectors based on your export format
            message_divs = soup.find_all('div', class_='chatlog__message')
            
            if not message_divs:
                # Try alternative Discord export formats
                message_divs = soup.find_all('div', attrs={'data-message-id': True})
            
            for msg_div in message_divs:
                # Extract timestamp
                timestamp_elem = msg_div.find('span', class_='chatlog__timestamp')
                if not timestamp_elem:
                    timestamp_elem = msg_div.find('time')
                
                # Extract message content
                content_elem = msg_div.find('div', class_='chatlog__content')
                if not content_elem:
                    content_elem = msg_div.find('span', class_='chatlog__markdown')
                
                if timestamp_elem and content_elem:
                    timestamp_str = timestamp_elem.get_text(strip=True)
                    message_text = content_elem.get_text(strip=True)
                    
                    # Try to parse timestamp
                    try:
                        # Common Discord export formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M %p', '%d-%b-%y %I:%M %p']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            logger.warning(f"Could not parse timestamp: {timestamp_str}")
                            continue
                            
                        messages.append((timestamp, message_text))
                    except Exception as e:
                        logger.warning(f"Error parsing message timestamp: {e}")
                        continue
                        
            logger.info(f"Extracted {len(messages)} messages from HTML export")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to parse HTML file: {e}", exc_info=True)
            return []

    def convert_to_backtest_format(self, messages, output_file, channel_name="exported"):
        """
        Converts parsed messages to signals_to_test.txt format.
        
        Args:
            messages: List of (timestamp, message_text) tuples
            output_file: Path to output signals_to_test.txt file
            channel_name: Name to use for the channel field
        """
        default_profile = self.config.profiles[0] if self.config.profiles else {
            'assume_buy_on_ambiguous': True,
            'ambiguous_expiry_enabled': True
        }
        
        valid_signals = []
        
        for timestamp, message_text in messages:
            # Try to parse as signal
            parsed_signal = self.signal_parser.parse_signal(message_text, default_profile)
            
            if parsed_signal:
                # Format: YYYY-MM-DD HH:MM:SS | channel_name | signal_text
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                line = f"{timestamp_str} | {channel_name} | {message_text}\n"
                valid_signals.append(line)
                logger.info(f"Parsed signal: {parsed_signal['ticker']} {parsed_signal['strike']}{parsed_signal['contract_type'][0]}")
            else:
                logger.debug(f"Message did not parse as signal: {message_text[:50]}...")
        
        # Write to file
        try:
            with open(output_file, 'w') as f:
                f.write("# Signals extracted from Discord HTML export\n")
                f.write("# Format: YYYY-MM-DD HH:MM:SS | channel_name | signal_text\n\n")
                f.writelines(valid_signals)
            
            logger.info(f"Successfully wrote {len(valid_signals)} valid signals to {output_file}")
            return len(valid_signals)
        except Exception as e:
            logger.error(f"Failed to write output file: {e}")
            return 0


async def main():
    """Main entry point for the HTML to signals converter."""
    script_dir = os.path.dirname(__file__)
    
    # Check for input HTML file
    html_file = os.path.join(script_dir, 'discord_export.html')
    if not os.path.exists(html_file):
        logger.error(f"No HTML file found at {html_file}")
        logger.info("Please place your Discord HTML export file as 'discord_export.html' in the backtester/ folder")
        logger.info("\nTo export Discord messages to HTML:")
        logger.info("1. Use a Discord export tool like DiscordChatExporter")
        logger.info("2. Export the channel to HTML format")
        logger.info("3. Save as 'discord_export.html' in this folder")
        return
    
    config = Config()
    converter = HTMLToSignals(config)
    
    # Parse HTML
    logger.info(f"Parsing HTML export from {html_file}...")
    messages = converter.parse_html_export(html_file)
    
    if not messages:
        logger.error("No messages extracted from HTML file")
        return
    
    # Convert to signals format
    output_file = os.path.join(script_dir, 'signals_to_test.txt')
    num_signals = converter.convert_to_backtest_format(messages, output_file)
    
    if num_signals > 0:
        logger.info(f"\n✅ SUCCESS! Converted {num_signals} signals")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Run: python backtester/data_harvester.py")
        logger.info(f"2. Run: python backtester/backtest_engine.py")
    else:
        logger.warning("No valid trading signals found in the HTML export")


if __name__ == "__main__":
    asyncio.run(main())
