import json
import logging
import os
from datetime import datetime
from ib_insync import Contract

class StateManager:
    """
    Manages the persistent state of the bot, including open positions and
    processed message IDs. Uses a human-readable JSON file.
    """
    def __init__(self, config):
        self.state_file_path = config.STATE_FILE_PATH
        # Ensure the directory for the state file exists
        os.makedirs(os.path.dirname(self.state_file_path), exist_ok=True)

    class ComplexEncoder(json.JSONEncoder):
        """Custom JSON encoder to handle complex objects like Contract and datetime."""
        def default(self, obj):
            if isinstance(obj, Contract):
                # Convert Contract object to a dictionary
                return {
                    '__type__': 'Contract',
                    'conId': obj.conId, 'symbol': obj.symbol, 'lastTradeDateOrContractMonth': obj.lastTradeDateOrContractMonth,
                    'strike': obj.strike, 'right': obj.right, 'exchange': obj.exchange, 'currency': obj.currency,
                    'localSymbol': obj.localSymbol, 'secType': obj.secType
                }
            if isinstance(obj, datetime):
                # Convert datetime object to ISO 8601 string format
                return {'__type__': 'datetime', 'isoformat': obj.isoformat()}
            return super().default(obj)

    def _object_hook(self, dct):
        """Custom JSON decoder to reconstruct complex objects."""
        if '__type__' in dct:
            if dct['__type__'] == 'Contract':
                # Reconstruct the Contract object from the dictionary
                return Contract(
                    conId=dct.get('conId'), symbol=dct.get('symbol'),
                    lastTradeDateOrContractMonth=dct.get('lastTradeDateOrContractMonth'),
                    strike=dct.get('strike'), right=dct.get('right'),
                    exchange=dct.get('exchange'), currency=dct.get('currency'),
                    localSymbol=dct.get('localSymbol'), secType=dct.get('secType')
                )
            if dct['__type__'] == 'datetime':
                # Reconstruct the datetime object from the ISO string
                return datetime.fromisoformat(dct['isoformat'])
        return dct

    def save_state(self, open_positions, processed_message_ids):
        """Saves the bot's current state to the JSON file."""
        state = {
            'open_positions': open_positions,
            'processed_message_ids': list(processed_message_ids)  # Convert deque to list for JSON
        }
        try:
            with open(self.state_file_path, 'w') as f:
                json.dump(state, f, cls=self.ComplexEncoder, indent=4)
            logging.info(f"Successfully saved state for {len(open_positions)} position(s).")
        except IOError as e:
            logging.error(f"Failed to save state file: {e}")

    def load_state(self):
        """
        Loads the bot's state from the JSON file.
        This function is guaranteed to return exactly two values.
        """
        if not os.path.exists(self.state_file_path):
            logging.warning(f"No state file found at {self.state_file_path}. Starting with a fresh state.")
            return {}, []  # Return two items: empty dict, empty list

        try:
            with open(self.state_file_path, 'r') as f:
                content = f.read()
                if not content:
                    logging.warning(f"State file {self.state_file_path} is empty. Starting with a fresh state.")
                    return {}, [] # Return two items

                data = json.loads(content, object_hook=self._object_hook)

                open_positions = data.get('open_positions', {})
                processed_message_ids = data.get('processed_message_ids', [])

                logging.info(f"Successfully loaded {len(open_positions)} open position(s) from state file.")
                # The primary return path. Returns exactly two items.
                return open_positions, processed_message_ids

        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Failed to load or parse state file: {e}. Starting with a fresh state.")
            return {}, [] # Return two items in case of error

