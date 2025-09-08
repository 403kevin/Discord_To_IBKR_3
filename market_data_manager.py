# services/market_data_manager.py
import logging
from threading import Thread, Lock

class MarketDataManager:
    """
    The "Master Watchmaker." This specialist's only job is to subscribe
    to live market data streams for active contracts and maintain a
    real-time price board. This is the definitive, battle-hardened
    solution for live price tracking.
    """
    def __init__(self, ib_interface):
        self.ib_interface = ib_interface
        self._price_board = {}
        self._lock = Lock()
        self._subscribed_contracts = set()

    def subscribe_to_contract(self, contract):
        """Subscribes to the live data stream for a new contract."""
        with self._lock:
            if contract.conId in self._subscribed_contracts:
                return # Already subscribed
            
            logging.info(f"[STREAM] Subscribing to market data for {contract.localSymbol}")
            self.ib_interface.ib.reqMktData(contract, '', False, False)
            self._subscribed_contracts.add(contract.conId)
            # Register the update handler
            self.ib_interface.ib.pendingTickersEvent += self._on_price_update

    def _on_price_update(self, tickers):
        """The event handler that updates the price board."""
        with self._lock:
            for ticker in tickers:
                self._price_board[ticker.contract.conId] = ticker

    def get_ticker(self, conId):
        """Gets the latest ticker data from the price board."""
        with self._lock:
            return self._price_board.get(conId)
