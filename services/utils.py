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