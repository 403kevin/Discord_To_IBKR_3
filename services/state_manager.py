import logging
import json
import os
from ib_insync import util as ib_util
from collections import deque

logger = logging.getLogger(__name__)


class StateManager:
    """
    A specialist module for saving and loading the bot's state.
    This is the "Amnesia Vaccine." It provides the bot with a persistent memory.
    """

    def __init__(self, state_file='active_state.json'):
        # The state file will be created in the main project directory.
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.state_file = os.path.join(project_root, state_file)

    def save_state(self, active_trades: dict, processed_ids: deque):
        """Saves the current active trades and processed IDs to a JSON file."""
        try:
            # We need to convert the complex ib_insync objects and deque into a storable format.
            state_to_save = {
                'active_trades': {},
                'processed_message_ids': list(processed_ids)
            }
            # ib_util.tree is the professional way to handle complex object serialization.
            for trade_id, trade_info in active_trades.items():
                state_to_save['active_trades'][trade_id] = {
                    'trade_obj': ib_util.tree(trade_info['trade_obj']),
                    'profile': trade_info['profile'],
                    'fill_processed': trade_info.get('fill_processed', False),
                    'entry_price': trade_info.get('entry_price'),
                    'sentiment_score': trade_info.get('sentiment_score', 'N/A'),
                    'high_water_mark': trade_info.get('high_water_mark', 0),
                    'native_trail_attached': trade_info.get('native_trail_attached', False),
                    'breakeven_armed': trade_info.get('breakeven_armed', False)
                }

            with open(self.state_file, 'w') as f:
                json.dump(state_to_save, f, indent=4, default=str)
            logger.info(f"Successfully saved state for {len(active_trades)} trades to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)

    def load_state(self, ib_instance):
        """
        Loads the state from the JSON file and reconstructs the trade objects.
        Returns a tuple of (active_trades, processed_message_ids).
        """
        if not os.path.exists(self.state_file):
            logger.info("No state file found. Starting with a fresh state.")
            return {}, []  # Return empty objects if no file exists

        try:
            with open(self.state_file, 'r') as f:
                loaded_state = json.load(f)

            rehydrated_trades = {}
            # Re-hydrate the trade objects from the saved data
            for trade_id, trade_data in loaded_state.get('active_trades', {}).items():
                # ib_util.tree_unflatten can perfectly reconstruct our trade objects
                trade_obj = ib_util.tree_unflatten(trade_data['trade_obj'])
                # We need to re-attach the live ib_insync instance for it to work
                trade_obj.contract.ib = ib_instance

                # Reconstruct the full trade_info dictionary
                rehydrated_trades[trade_id] = trade_data
                rehydrated_trades[trade_id]['trade_obj'] = trade_obj

            processed_ids = loaded_state.get('processed_message_ids', [])

            logger.info(f"Successfully loaded state with {len(rehydrated_trades)} active trades.")
            return rehydrated_trades, processed_ids
        except Exception as e:
            logger.error(f"Failed to load state from {self.state_file}: {e}. Starting fresh.", exc_info=True)
            return {}, []

