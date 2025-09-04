from datetime import datetime, timedelta

def get_next_friday() -> datetime:
    """
    Calculates the date of the upcoming Friday.
    If today is Friday, it returns today's date.
    """
    today = datetime.today()
    # weekday() returns 0 for Monday and 4 for Friday.
    # The calculation is: 4 (Friday) - today's weekday number.
    # If today is Monday (0), days_to_friday is 4.
    # If today is Thursday (3), days_to_friday is 1.
    # If today is Friday (4), days_to_friday is 0.
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
