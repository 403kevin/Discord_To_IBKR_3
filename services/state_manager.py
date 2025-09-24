import json
import logging
import os
import tempfile
from datetime import datetime
from ib_insync import Contract, Stock, Option

# ==============================================================================
# --- JSON ENCODER & DECODER FOR COMPLEX IBKR/DATETIME OBJECTS ---
# ==============================================================================

class ComplexEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle complex objects like datetime and IBKR Contracts.
    """
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, Contract):
            # Convert the Contract object to a dictionary
            return {"__contract__": {
                "conId": obj.conId,
                "symbol": obj.symbol,
                "secType": obj.secType,
                "lastTradeDateOrContractMonth": obj.lastTradeDateOrContractMonth,
                "strike": obj.strike,
                "right": obj.right,
                "multiplier": obj.multiplier,
                "exchange": obj.exchange,
                "currency": obj.currency,
                "localSymbol": obj.localSymbol,
                "tradingClass": obj.tradingClass
            }}
        return super().default(obj)

def as_complex_object(dct):
    """
    Custom JSON decoder to reconstruct complex objects from our dictionary format.
    """
    if "__datetime__" in dct:
        return datetime.fromisoformat(dct["__datetime__"])
    if "__contract__" in dct:
        # Reconstruct the Contract object from the dictionary
        contract_data = dct["__contract__"]
        # Determine if it's a Stock or Option and create the appropriate object
        if contract_data['secType'] == 'OPT':
            return Option(**contract_data)
        elif contract_data['secType'] == 'STK':
            return Stock(**contract_data)
        else: # Fallback for other contract types
            return Contract(**contract_data)
    return dct

# ==============================================================================
# --- STATE MANAGER CLASS ---
# ==============================================================================

class StateManager:
    """
    Handles the saving and loading of the bot's state to a JSON file.
    This includes open positions, cooldown timestamps, and other persistent data.
    Uses a robust write-and-rename protocol to prevent data corruption.
    """
    def __init__(self, config):
        self.config = config
        self.state_file_path = "open_positions.json"
        self.state = self.load_state()

    def load_state(self):
        """
        Loads the bot's state from the JSON file.
        If the file doesn't exist or is empty, returns a default state.
        """
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    state_data = json.load(f, object_hook=as_complex_object)
                    logging.info("Successfully loaded state from %s.", self.state_file_path)
                    return state_data
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.error("Error loading state file %s: %s. Starting with a fresh state.", self.state_file_path, e)
                return self._get_default_state()
        else:
            logging.warning("No state file found at %s. Starting with a fresh state.", self.state_file_path)
            return self._get_default_state()

    def save_state(self):
        """
        Saves the current state to the JSON file using a safe write-and-rename protocol.
        This prevents data corruption if the bot is terminated mid-write.
        """
        try:
            # Create a temporary file in the same directory
            temp_file_path = ""
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.path.dirname(self.state_file_path), suffix='.tmp') as temp_f:
                temp_file_path = temp_f.name
                json.dump(self.state, temp_f, cls=ComplexEncoder, indent=4)

            # Atomically rename the temporary file to the final state file
            os.replace(temp_file_path, self.state_file_path)
            logging.debug("Successfully saved state to %s.", self.state_file_path)
        except Exception as e:
            logging.error("CRITICAL: Failed to save state! Error: %s", e)
            # Attempt to clean up the temp file if it still exists
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def get_open_positions(self):
        """Returns the list of open positions from the state."""
        return self.state.get("open_positions", [])

    def add_position(self, position):
        """Adds a new position to the state and saves."""
        self.state["open_positions"].append(position)
        self.save_state()
        logging.info("Added new position to state: %s", position)

    def remove_position(self, conId):
        """Removes a position from the state by its contract ID (conId) and saves."""
        initial_count = len(self.state["open_positions"])
        self.state["open_positions"] = [p for p in self.state["open_positions"] if p['contract'].conId != conId]
        if len(self.state["open_positions"]) < initial_count:
            self.save_state()
            logging.info("Removed position with conId %s from state.", conId)
        else:
            logging.warning("Attempted to remove position with conId %s, but it was not found in state.", conId)

    def update_cooldown(self, channel_id):
        """Updates the cooldown timestamp for a given channel and saves."""
        self.state["channel_cooldowns"][channel_id] = datetime.now()
        self.save_state()

    def get_cooldown(self, channel_id):
        """Gets the cooldown timestamp for a given channel."""
        return self.state["channel_cooldowns"].get(channel_id)

    def _get_default_state(self):
        """Returns the default structure for a new state."""
        return {
            "open_positions": [],
            "channel_cooldowns": {},
            "consecutive_losses": {} # Placeholder for potential future use
        }