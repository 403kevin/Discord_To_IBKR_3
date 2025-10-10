"""
services/tickers.py

Master whitelist of valid stock tickers for signal parsing.
Only tickers in this file will be recognized as valid symbols.

STRICT MODE: Any ticker not in this list will be REJECTED.
"""

# =============================================================================
# CUSTOM TICKERS - Add your own here
# =============================================================================
# Use this section for tickers you want to trade that aren't in the lists below.
# Example: CUSTOM = {'RIVN', 'LCID', 'PLTR', 'SOFI'}

CUSTOM = set()


# =============================================================================
# MAJOR INDICES (SPX, SPY, etc.)
# =============================================================================
INDICES = {
    'SPX',      # S&P 500 Index
    'SPXW',     # S&P 500 Weekly Index
    'SPY',      # SPDR S&P 500 ETF
    'NDX',      # NASDAQ-100 Index
    'DJX',      # Dow Jones Index
    'RUT',      # Russell 2000 Index
    'VIX',      # Volatility Index
}


# =============================================================================
# MAJOR ETFs (High Volume, Commonly Traded Options)
# =============================================================================
MAJOR_ETFS = {
    # Broad Market
    'SPY',      # S&P 500
    'QQQ',      # NASDAQ-100
    'IWM',      # Russell 2000
    'DIA',      # Dow Jones
    'VOO',      # Vanguard S&P 500
    'VTI',      # Vanguard Total Market
    
    # Sector ETFs
    'XLF',      # Financials
    'XLE',      # Energy
    'XLK',      # Technology
    'XLV',      # Healthcare
    'XLI',      # Industrials
    'XLP',      # Consumer Staples
    'XLY',      # Consumer Discretionary
    'XLU',      # Utilities
    'XLB',      # Materials
    'XLRE',     # Real Estate
    'XLC',      # Communications
    
    # Volatility
    'VXX',      # Short-Term VIX Futures
    'UVXY',     # 2x VIX Short-Term
    'SVXY',     # Inverse VIX Short-Term
    
    # Tech-Heavy
    'TQQQ',     # 3x NASDAQ-100
    'SQQQ',     # 3x Inverse NASDAQ-100
    'TECL',     # 3x Technology
    
    # Gold/Silver
    'GLD',      # Gold
    'SLV',      # Silver
    'GDX',      # Gold Miners
    
    # Oil/Energy
    'USO',      # Oil Fund
    'XLE',      # Energy Sector
    'XOP',      # Oil & Gas Exploration
    
    # Bonds
    'TLT',      # 20+ Year Treasury
    'IEF',      # 7-10 Year Treasury
    'SHY',      # 1-3 Year Treasury
    
    # International
    'EEM',      # Emerging Markets
    'EFA',      # EAFE (Europe, Asia, Far East)
    'FXI',      # China Large-Cap
}


# =============================================================================
# S&P 500 COMPANIES (Auto-Generated - Current as of 2025)
# =============================================================================
SP500 = {
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CSCO', 'ADBE', 'CRM', 'ACN', 'AMD',
    'IBM', 'INTC', 'QCOM', 'TXN', 'INTU', 'NOW', 'AMAT', 'MU', 'LRCX', 'KLAC',
    'SNPS', 'CDNS', 'MCHP', 'ADI', 'FTNT', 'ANSS', 'PLTR', 'PANW', 'ANET', 'APH',
    'TEL', 'MSI', 'HPQ', 'NTAP', 'STX', 'WDC', 'ZBRA', 'EPAM', 'FFIV', 'JNPR',
    
    # Communication Services
    'GOOGL', 'GOOG', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
    'EA', 'TTWO', 'MTCH', 'NWSA', 'NWS', 'FOXA', 'FOX', 'LYV', 'OMC', 'IPG',
    
    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'AZO',
    'CMG', 'MAR', 'GM', 'F', 'ABNB', 'ORLY', 'HLT', 'YUM', 'DHI', 'LEN',
    'ROST', 'GRMN', 'EBAY', 'EXPE', 'POOL', 'TPR', 'RL', 'ULTA', 'DPZ', 'DRI',
    'GPC', 'BBY', 'MHK', 'WHR', 'LVS', 'WYNN', 'MGM', 'CZR', 'TSCO', 'AAP',
    
    # Consumer Staples
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
    'GIS', 'KHC', 'HSY', 'K', 'CAG', 'SJM', 'CPB', 'MKC', 'CHD', 'CLX',
    'TSN', 'HRL', 'LW', 'TAP', 'STZ', 'BF.B', 'DG', 'DLTR', 'KR', 'SYY',
    
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB',
    'KMI', 'HAL', 'BKR', 'HES', 'DVN', 'FANG', 'MRO', 'APA', 'CTRA', 'EQT',
    
    # Financials
    'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
    'C', 'AXP', 'CB', 'MMC', 'PGR', 'TFC', 'USB', 'PNC', 'BK', 'TRV',
    'AON', 'AFL', 'ALL', 'MET', 'PRU', 'AIG', 'COF', 'AJG', 'HIG', 'CINF',
    'CME', 'ICE', 'MCO', 'MSCI', 'NDAQ', 'SCHW', 'TROW', 'BEN', 'IVZ', 'NTRS',
    
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY',
    'AMGN', 'CVS', 'ELV', 'GILD', 'CI', 'ISRG', 'REGN', 'VRTX', 'MCK', 'HCA',
    'BSX', 'MDT', 'SYK', 'ZTS', 'BDX', 'EW', 'A', 'IDXX', 'HUM', 'CNC',
    'IQV', 'RMD', 'DXCM', 'MTD', 'BAX', 'ALGN', 'HOLX', 'WAT', 'STE', 'TECH',
    
    # Industrials
    'CAT', 'RTX', 'HON', 'UPS', 'BA', 'LMT', 'GE', 'DE', 'UNP', 'ADP',
    'MMM', 'GD', 'NOC', 'ETN', 'ITW', 'WM', 'EMR', 'CSX', 'NSC', 'FDX',
    'PCAR', 'TT', 'PH', 'CMI', 'ROK', 'CARR', 'OTIS', 'JCI', 'FAST', 'PAYX',
    'URI', 'RSG', 'ODFL', 'VRSK', 'IEX', 'XYL', 'NDSN', 'DOV', 'FTV', 'BR',
    
    # Materials
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL', 'DD', 'DOW', 'PPG', 'NUE',
    'CTVA', 'VMC', 'MLM', 'ALB', 'BALL', 'AVY', 'AMCR', 'PKG', 'IP', 'CE',
    
    # Real Estate
    'PLD', 'AMT', 'EQIX', 'PSA', 'SPG', 'WELL', 'DLR', 'O', 'CBRE', 'EXR',
    'AVB', 'EQR', 'VICI', 'VTR', 'SBAC', 'WY', 'INVH', 'MAA', 'ARE', 'DOC',
    
    # Utilities
    'NEE', 'SO', 'DUK', 'CEG', 'SRE', 'AEP', 'VST', 'D', 'PEG', 'EXC',
    'XEL', 'ED', 'WEC', 'AWK', 'ES', 'DTE', 'FE', 'ETR', 'PPL', 'EIX',
}


# =============================================================================
# POPULAR OPTIONS PLAYS (Not in S&P 500 but heavily traded)
# =============================================================================
POPULAR_OPTIONS = {
    # Mega-Cap Tech (some overlap with S&P 500)
    'TSLA', 'NVDA', 'AMD', 'META', 'GOOGL', 'AAPL', 'MSFT', 'AMZN',
    
    # Growth/Meme Stocks
    'PLTR', 'SOFI', 'HOOD', 'COIN', 'RBLX', 'RIVN', 'LCID', 'NIO',
    'XPEV', 'LI', 'GRAB', 'DKNG', 'PENN', 'SHOP', 'SQ', 'PYPL',
    
    # Biotech/Pharma
    'MRNA', 'BNTX', 'SAVA', 'NVAX', 'VXRT',
    
    # SPACs/Recent IPOs
    'DASH', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'NET', 'MDB', 'OKTA',
    
    # Retail/Consumer
    'GME', 'AMC', 'BBBY', 'BYND', 'PTON', 'LULU', 'CHWY',
    
    # Crypto-Related
    'MARA', 'RIOT', 'BTBT', 'MSTR', 'CLSK', 'CAN', 'HUT', 'BITF',
    
    # EV/Clean Energy
    'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'FSR', 'RIDE',
    
    # Chinese Stocks
    'BABA', 'JD', 'PDD', 'BIDU', 'BILI', 'NTES',
}


# =============================================================================
# MASTER WHITELIST (All Combined)
# =============================================================================
VALID_TICKERS = CUSTOM | INDICES | MAJOR_ETFS | SP500 | POPULAR_OPTIONS


# =============================================================================
# HELPER FUNCTION (Optional - for debugging)
# =============================================================================
def is_valid_ticker(ticker: str) -> bool:
    """
    Check if a ticker is in the whitelist.
    
    Args:
        ticker: Stock symbol to validate
        
    Returns:
        True if ticker is valid, False otherwise
    """
    return ticker.upper() in VALID_TICKERS


def get_ticker_count() -> dict:
    """
    Get count of tickers by category.
    
    Returns:
        Dictionary with counts per category
    """
    return {
        'custom': len(CUSTOM),
        'indices': len(INDICES),
        'major_etfs': len(MAJOR_ETFS),
        'sp500': len(SP500),
        'popular_options': len(POPULAR_OPTIONS),
        'total': len(VALID_TICKERS)
    }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    print("="*80)
    print("TICKER WHITELIST STATISTICS")
    print("="*80)
    
    stats = get_ticker_count()
    for category, count in stats.items():
        print(f"{category.upper()}: {count}")
    
    print("\n" + "="*80)
    print("TESTING VALIDATION")
    print("="*80)
    
    test_tickers = ['SPY', 'AAPL', 'SIZED', 'PLACE', 'BELOW', 'TSLA', 'FAKE']
    for ticker in test_tickers:
        result = "✅ VALID" if is_valid_ticker(ticker) else "❌ INVALID"
        print(f"{ticker}: {result}")
