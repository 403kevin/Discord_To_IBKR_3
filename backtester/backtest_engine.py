"""
FIXED VERSION: Universal Column Handler
Automatically detects and normalizes 'timestamp' OR 'ts_event' column names
"""

# Just the _load_signal_data method - paste this into your backtest_engine.py
# to replace the existing _load_signal_data method

def _load_signal_data(self, signal: Dict) -> pd.DataFrame:
    """
    Load historical tick data for the signal with UNIVERSAL COLUMN HANDLING.
    
    CRITICAL FIX: Auto-detects 'timestamp' or 'ts_event' columns and normalizes them.
    """
    ticker = signal['ticker']
    expiry = signal['expiry']
    strike = signal['strike']
    right = signal['right']
    
    # Generate the filename
    filename = get_data_filename_databento(ticker, expiry, strike, right)
    filepath = os.path.join(self.historical_data_dir, filename)
    
    if not os.path.exists(filepath):
        logging.warning(f"Historical data not found: {filepath}")
        return pd.DataFrame()
    
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # ===== UNIVERSAL COLUMN HANDLER FIX =====
        # Check which timestamp column exists and normalize to 'ts_event'
        if 'timestamp' in df.columns and 'ts_event' not in df.columns:
            # Rename 'timestamp' to 'ts_event' for consistency
            df = df.rename(columns={'timestamp': 'ts_event'})
            logging.debug(f"✅ Normalized 'timestamp' → 'ts_event' for {filename}")
        elif 'ts_event' not in df.columns and 'timestamp' not in df.columns:
            # Neither column exists - this is a problem
            logging.error(f"❌ No timestamp column found in {filename}. Columns: {df.columns.tolist()}")
            return pd.DataFrame()
        # If 'ts_event' already exists, we're good - no action needed
        
        # Ensure ts_event is datetime
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        
        # Filter to only include data from signal time onwards
        signal_time = pd.to_datetime(signal['timestamp'])
        df = df[df['ts_event'] >= signal_time].copy()
        
        # Sort by timestamp
        df = df.sort_values('ts_event').reset_index(drop=True)
        
        # Add signal reference
        df['signal_id'] = signal.get('id', 'unknown')
        
        logging.info(f"✅ Loaded {len(df)} ticks for {ticker} {strike}{right}")
        return df
        
    except Exception as e:
        logging.error(f"❌ Error loading data from {filepath}: {e}")
        return pd.DataFrame()
