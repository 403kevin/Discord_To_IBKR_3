from ib_insync import Contract

def get_data_filename(contract: Contract) -> str:
    """
    The single, non-negotiable source of truth for generating data filenames.
    Both the harvester and the mock interface MUST call this function.
    This permanently eliminates the risk of a filename mismatch.

    Args:
        contract: The IBKR Contract object.

    Returns:
        A standardized filename string.
    """
    # Example Output: SPX_20250926_6640C_5sec_data.csv
    symbol = contract.symbol
    expiry = contract.lastTradeDateOrContractMonth
    strike = int(contract.strike)
    right = contract.right[0].upper() # Just 'C' or 'P'

    return f"{symbol}_{expiry}_{strike}{right}_5sec_data.csv"

def get_data_filename_databento(ticker, expiry, strike, right):
    """
    Generate filename for Databento-sourced data.
    
    Args:
        ticker: Stock symbol
        expiry: Expiry date string (YYYYMMDD format)
        strike: Strike price (int or float)
        right: 'C' or 'P'
    
    Returns:
        Standardized filename
    """
    # Convert strike to float, then check if it's a whole number
    strike_float = float(strike)
    
    # If it's a whole number (like 205.0), convert to int to remove .0
    # If it's a decimal (like 10.5), keep the decimal
    if strike_float == int(strike_float):
        strike_str = str(int(strike_float))
    else:
        strike_str = str(strike_float)
    
    return f"{ticker}_{expiry}_{strike_str}{right}_databento.csv"
