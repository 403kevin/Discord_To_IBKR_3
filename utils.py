# =============================================================================================== #
from datetime import datetime, timedelta
import config

# =============================================================================================== #

def get_next_friday(symbol=''):
    days_to_str = {"monday": 4, "tuesday": 3, "wednesday": 2, "thursday": 1, "friday": 0}
    inc = days_to_str[datetime.today().strftime('%A').lower()]
    if config.NEXT_FRIDAY_IS_A_HOLIDAY:
        inc -= 1
    if symbol == 'spx':
        inc = 0
    day = str((datetime.now() + timedelta(inc)).day)
    day = day if len(day) == 2 else '0' + day
    month = str((datetime.now() + timedelta(inc)).month)
    month = month if len(month) == 2 else '0' + month
    dtt = str(datetime.now().year) + month + day
    return dtt

def get_business_day(dte: int) -> datetime:
    """Calculate expiry date skipping weekends"""
    today = datetime.today()
    target_date = today + timedelta(days=dte)
    # Skip weekends (Saturday=5, Sunday=6)
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    return target_date

# =============================================================================================== #