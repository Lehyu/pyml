import pandas as pd
import datetime


def load_data(path, cols=None, date_format='%Y-%m-%d %H'):
    if cols is not None:
        data = pd.read_csv(path, parse_dates=cols, date_parser=lambda dates: pd.datetime.strptime(dates, date_format))
    else:
        data = pd.read_csv(path)
    return data


def extract_data(data, condition, cols=None):
    if cols is None:
        return data.loc[condition, :]
    else:
        return data.loc[condition, cols]


def add_statistic_feature(data, candidate_cols, key_cols, name, option="mean"):
    com_cols = [col for col in key_cols]
    com_cols.extend(candidate_cols)
    if option == "mean":
        _df = data[com_cols].groupby(key_cols).mean().reset_index(level=key_cols)
    elif option == "median":
        _df = data[com_cols].groupby(key_cols).median().reset_index(level=key_cols)
    elif option == "max":
        _df = data[com_cols].groupby(key_cols).max().reset_index(level=key_cols)
    elif option == "min":
        _df = data[com_cols].groupby(key_cols).min().reset_index(level=key_cols)
    else:
        raise ValueError("Option %s hasn't been supported yet!" % option)
    if len(_df) == 0:
        print(name)
        return data
    postfix = '_' + str(name) + '_' + option
    _df = _df.rename(columns=lambda col: col + postfix if col in candidate_cols else col)
    data = pd.merge(data, _df, on=key_cols)
    return data


def append_statistic_feature(to_df, from_df, candidate_cols, key_cols, name, option="mean"):
    com_cols = [col + '_' + name + '_' + option for col in candidate_cols]
    com_cols.extend(key_cols)
    df = from_df[com_cols]
    df = df.drop_duplicates()
    to_df = pd.merge(to_df, df, on=key_cols, how='left')
    return to_df


def shift_time(date, hours=1, left=True):
    if left:
        return date - datetime.timedelta(hours=hours)
    else:
        return date + datetime.timedelta(hours=hours)
