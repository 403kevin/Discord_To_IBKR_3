import datetime

# Discord's "epoch" is the first second of the year 2015.
DISCORD_EPOCH = 1420070400000

def snowflake_to_datetime(snowflake: int) -> datetime.datetime:
    """
    Converts a Discord Snowflake ID into a precise, UTC datetime object.
    """
    # The timestamp is the first 42 bits of the snowflake ID.
    timestamp_ms = (snowflake >> 22) + DISCORD_EPOCH
    # Convert from milliseconds to a datetime object.
    return datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezone.utc)

def main():
    """
    A simple tool to convert a Discord Message ID into a backtest-ready timestamp.
    """
    print("--- Discord Snowflake to Timestamp Converter ---")
    print("Right-click a message in Discord (with Developer Mode on) and 'Copy Message ID'.")
    
    while True:
        try:
            snowflake_id_str = input("Paste Message ID here (or 'q' to quit): ")
            if snowflake_id_str.lower() == 'q':
                break
                
            snowflake_id = int(snowflake_id_str)
            
            # Use our function to decrypt the ID
            precise_datetime = snowflake_to_datetime(snowflake_id)
            
            # Format it perfectly for our signals_to_test.txt file
            formatted_timestamp = precise_datetime.strftime('%Y-%m-%d %H:%M:%S')
            
            print("\n" + "="*30)
            print(f"✅ Backtest Timestamp: {formatted_timestamp}")
            print("="*30 + "\n")

        except ValueError:
            print("❌ Invalid input. Please paste a valid, numeric Message ID.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()