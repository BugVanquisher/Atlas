import datetime as dt


def ymd_now():
    now = dt.datetime.now(dt.UTC)
    return now.strftime("%Y%m%d"), now.strftime("%Y%m")