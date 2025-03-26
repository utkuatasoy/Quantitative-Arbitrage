# data_reader.py
import pandas as pd

def load_data(file_path, date_col="Date", dayfirst=True):
    data = pd.read_csv(file_path, parse_dates=[date_col], dayfirst=dayfirst)
    data.set_index(date_col, inplace=True)
    data.index = pd.to_datetime(data.index)
    return data