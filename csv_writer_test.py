import ccxt
import csv
import logging
import os
import time
from pytz import timezone
from datetime import datetime

def write_to_csv(row_dict):
    """
    Adds a row to a CSV file. If the file does not exist, it creates one with the specified column names.

    :param file_path: Path to the CSV file.
    :param column_names: List of column names for the CSV.
    :param row_dict: A dictionary representing the row to be added, with keys as column names.
    """

    file_path = 'trading.cvs'
    column_names = ['datetime', 'symbol', 'indicator', 'order_type', 'price', 'amount']
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)

        # If the file doesn't exist or is empty, write the header
        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()

        # Write the row data
        writer.writerow(row_dict)

def current_time():
    KST = timezone('Asia/Seoul')
    datetime_now = datetime.now()
    localtime_now = datetime_now.astimezone(KST)
    return localtime_now

def save_data(symbol, indicator, order_type, price, amount):
    datetime_now = current_time()
    csv_row = {
             'datetime': datetime_now,
             'symbol': symbol,
             'indicator': indicator,
             'order_type': order_type,
             'price' : round(price, 1),
             'amount': round(amount, 0),
    }
    write_to_csv( csv_row )

save_data("DOGE/KRW", "CCI", "Buy", 100, 1000)
