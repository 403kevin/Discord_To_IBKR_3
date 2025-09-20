import pandas as pd
import pandas_ta as ta

class TechnicalAnalyzer:
    """
    The "Flight Computer." A specialist module for calculating technical
    indicators from historical price data.
    """
    def __init__(self):
        pass

    def calculate_indicators(self, price_data: pd.DataFrame, profile: dict):
        """
        Takes a dataframe of historical prices and the trade's profile,
        and returns a dictionary of calculated indicator values.
        """
        if price_data.empty:
            return {}

        indicators = {}
        exit_strategy = profile['exit_strategy']

        # Calculate ATR if needed
        if exit_strategy['trail_method'] == 'atr':
            atr_settings = exit_strategy['trail_settings']
            atr_df = ta.atr(price_data['high'], price_data['low'], price_data['close'], length=atr_settings['atr_period'])
            if atr_df is not None and not atr_df.empty:
                indicators['atr'] = atr_df.iloc[-1]

        # Calculate PSAR if needed
        if exit_strategy['momentum_exits']['psar_enabled']:
            psar_settings = exit_strategy['momentum_exits']['psar_settings']
            psar_df = ta.psar(price_data['high'], price_data['low'], af0=psar_settings['start'], af=psar_settings['increment'], max_af=psar_settings['max'])
            if psar_df is not None and not psar_df.empty:
                # We need the PSAR long and short values to check for flips
                indicators['psar_long'] = psar_df['PSARl_0.02_0.2'].iloc[-1]
                indicators['psar_short'] = psar_df['PSARs_0.02_0.2'].iloc[-1]

        # Calculate RSI if needed
        if exit_strategy['momentum_exits']['rsi_hook_enabled']:
            rsi_settings = exit_strategy['momentum_exits']['rsi_settings']
            rsi_series = ta.rsi(price_data['close'], length=rsi_settings['period'])
            if rsi_series is not None and not rsi_series.empty:
                indicators['rsi'] = rsi_series.iloc[-1]

        return indicators