import pandas as pd
import datetime

def load_data(path, cols=[], date_format='%Y-%m-%d %H'):
    if len(cols) != 0:
        data = pd.read_csv(path, parse_dates=cols, date_parser=lambda dates: pd.datetime.strptime(dates, date_format))
    else:
        data = pd.read_csv(path)
    return data


def extract_data(data, condition, cols=None):
    if cols is None:
        return data.loc[condition, :]
    else:
        return data.loc[condition, cols]

def shif_time(date, hours=1, left=True):
    if left:
        return date-datetime.timedelta(hours=hours)
    else:
        return date+datetime.timedelta(hours=hours)