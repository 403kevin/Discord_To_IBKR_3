# services/utils.py
import logging
from datetime import datetime, timedelta

# This file holds various helper functions used across the application.
# The problematic screen-clearing function has been surgically removed
# to resolve the 'posix' error, while restoring the essential date
# calculation logic.

def get_next_friday() -> datetime:
    """
    Calculates the date of the upcoming Friday.
    If today is Friday, it returns today's date.
    """
    today = datetime.today()
    # weekday() returns 0 for Monday and 4 for Friday.
    days_to_friday = (4 - today.weekday() + 7) % 7
    next_friday = today + timedelta(days=days_to_friday)
    return next_friday

def get_business_day(dte: int) -> datetime:
    """
    Calculates a future business day by adding a number of days to today,
    skipping any weekends.
    """
    target_date = datetime.today() + timedelta(days=dte)
    # weekday() returns 5 for Saturday and 6 for Sunday.
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    return target_date

logging.info("Universal utils module loaded correctly.")

