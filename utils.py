from datetime import datetime, timedelta

def get_business_day(dte: int) -> datetime:
    """
    Calculates a future business day by adding a number of days to today,
    skipping any weekends.

    Args:
        dte (int): The number of days to expiry. For example, 0 for today,
                   1 for the next business day, etc.

    Returns:
        A datetime object representing the target expiry date.
    """
    # Start with today's date
    target_date = datetime.today() + timedelta(days=dte)

    # If the target date lands on a weekend, move it to the next Monday.
    # Note: weekday() returns 5 for Saturday and 6 for Sunday.
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
        
    return target_date

# The old `get_next_friday` function has been removed. It was complex and
# less reliable than handling explicit dates or DTE formats, which our
# new parser is designed to do. The `get_business_day` function is a
# much more robust way to handle relative expiries.

