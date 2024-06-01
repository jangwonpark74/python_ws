import ccxt
import csv
import logging
import os
import pandas as pd
import pandas_ta as ta
import schedule
import talib
import time
import random

from conf import key
from collections import defaultdict
from pprint import pprint
from pytz import timezone
from datetime import datetime

# Decision for orders 
mfi_sell_decision= defaultdict(bool)
cci_buy_decision = defaultdict(bool)
stochrsi_buy_decision = defaultdict(bool)
supertrend_sell_decision= defaultdict(bool)

# Current CCI 
current_cci = defaultdict(float)

# MFI(14), 4 hour based weight control
mfi_weight = defaultdict(float)

# Order amount
# MFI amount will be multiplied by MFI weight
mfi_sell_amount = 2000000
mfi_buy_amount  = 2000000
cci_buy_amount  = 3000000
stochrsi_buy_amount  = 3000000

# 4 Hour supertrend order amount 
supertrend_sell_amount = 10000000

# Threshold for each trading strategy
cci_low_threshold = -150
mfi_high_threshold = 80
stochrsi_low_threshold = 25

# Pullback stratey 
pullback_portion = 0.6

pd.set_option('display.max_rows', None)

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

def save_trading_data(symbol, indicator, order_type, price, amount):
    datetime_now = current_time()
    csv_row = {
             'datetime': datetime_now,
             'symbol': symbol,
             'indicator': indicator,
             'order_type': order_type,
             'price' : round(price, 1),
             'amount': round(amount, 0),
    }
    write_to_cvs( csv_row )

def show_orderbook(orderbook):
    print("\n------------Getting order book -----------")
    pprint(orderbook)

def calc_mfi_amount(symbol):
    amount = mfi_sell_amount * mfi_weight[symbol]
    return amount

def analyze_stochrsi_signal(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='10m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        stochrsi = ta.stochrsi(df['close'], length=14, fastk_period=3, fastd_period=3, append=True)

        df['stochrsi_k'] = stochrsi['STOCHRSIk_14_14_3_3']
        df['stochrsi_d'] = stochrsi['STOCHRSId_14_14_3_3']

        # Get the latest value
        current_stochrsi_k = df['stochrsi_k'].iloc[-1]
        current_stochrsi_d = df['stochrsi_d'].iloc[-1]

        # Stoch rsi cross-over strategy
        buy  = (current_stochrsi_k > current_stochrsi_d) and (current_stochrsi_k < stochrsi_low_threshold)

        global stochrsi_buy_decision
        stochrsi_buy_decision[symbol] = buy

        df['stochrsi_buy']  = buy

        print(f'\n----------- {symbol} STOCHRSI Signal Analysis (10 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_mfi_signal(exchange, symbol: str)->None:
    try:
        ohlcv_5m = exchange.fetch_ohlcv(symbol, timeframe='5m')
        df = pd.DataFrame(ohlcv_5m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi_5m'] = round(talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 1)

        mfi_5m = df['mfi_5m'].iloc[-1]
        # store information for dispaly

        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h')
        df_1h = pd.DataFrame(ohlcv_1h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['datetime'] = pd.to_datetime(df_1h['datetime'], utc=True, unit='ms')
        df_1h['datetime'] = df_1h['datetime'].dt.tz_convert("Asia/Seoul")
        df_1h['mfi_1h'] = round(talib.MFI(df_1h['high'], df_1h['low'], df_1h['close'], df_1h['volume'], timeperiod=14), 1)

        mfi_1h = df_1h['mfi_1h'].iloc[-1]

        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df_4h = pd.DataFrame(ohlcv_4h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df_4h['datetime'] = df_4h['datetime'].dt.tz_convert("Asia/Seoul")
        df_4h['mfi_4h'] = round(talib.MFI(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume'], timeperiod=14), 1)

        # Scalping based on MFI and RSI every 4 hours
        mfi_4h = df_4h['mfi_4h'].iloc[-1]

        mfi = (mfi_5m + mfi_1h + mfi_4h)/3.0
        global mfi_weight
        mfi_weight[symbol] = round(abs(mfi - 50)/20.0, 1)

        print(f'\n----------- {symbol} MFI Signal Analysis (5 minutes) --------------')
        pprint(df.iloc[-1])
        print(f'\n----------- {symbol} MFI Signal Analysis ( 1 hour) --------------')
        pprint(df_1h.iloc[-1])
        print(f'\n----------- {symbol} MFI Signal Analysis (4 hours) --------------')
        pprint(df_4h.iloc[-1])

        # current cci 
        cci = current_cci[symbol]
        cci_factor = cci / 140.0

        # update data for execution of order
        global mfi_sell_decision
        mfi_sell_decision[symbol] = (mfi*cci_factor) > mfi_high_threshold 

    except Exception as e:
        print("Exception : ", str(e))

def analyze_cci_signal(exchange, symbol: str)->None:
    try:
        ohlcv_5m = exchange.fetch_ohlcv(symbol, timeframe='5m')
        df = pd.DataFrame(ohlcv_5m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['cci_5m']   = round(talib.CCI(df['high'], df['low'], df['close'], timeperiod=20), 1)

        cci_5m = df['cci_5m'].iloc[-1]
        # store information for dispaly

        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h')
        df_1h = pd.DataFrame(ohlcv_1h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['datetime'] = pd.to_datetime(df_1h['datetime'], utc=True, unit='ms')
        df_1h['datetime'] = df_1h['datetime'].dt.tz_convert("Asia/Seoul")
        df_1h['cci_1h']   = round(talib.CCI(df_1h['high'], df_1h['low'], df_1h['close'], timeperiod=20), 1)

        cci_1h = df_1h['cci_1h'].iloc[-1]

        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df_4h = pd.DataFrame(ohlcv_4h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df_4h['datetime'] = df_4h['datetime'].dt.tz_convert("Asia/Seoul")
        df_4h['cci_4h']   = round(talib.CCI(df_4h['high'], df_4h['low'], df_4h['close'], timeperiod=20), 1)

        # Scalping based on MFI and RSI every 4 hours
        cci_4h = df_4h['cci_4h'].iloc[-1]
        cci = (cci_5m + cci_1h + cci_4h)/3.0

        global current_cci
        current_cci[symbol] = cci

        global cci_buy_decision
        cci_buy_decision[symbol] = (cci < cci_low_threshold) 

        print(f'\n----------- {symbol} CCI Signal Analysis (5 minutes) --------------')
        pprint(df.iloc[-1])
        print(f'\n----------- {symbol} CCI Signal Analysis ( 1 hour) --------------')
        pprint(df_1h.iloc[-1])
        print(f'\n----------- {symbol} CCI Signal Analysis (4 hours) --------------')
        pprint(df_4h.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_supertrend_signal(exchange, symbol: str)->None:
    try:
        # upto two weeks analyze supertrend 
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        period = 14
        multiplier = 2.0

        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

        df['upperband'] = (df['high'] + df['low'])/2 + (multiplier * df['atr'])
        df['lowerband'] = (df['high'] + df['low'])/2 - (multiplier * df['atr'])

        df['in_uptrend'] = False

        for i in range(1, len(df.index)):
            p = i - 1

            if df['close'][i] > df['upperband'][p]:
                df.loc[i, 'in_uptrend'] = True
            elif df['close'][i] < df['lowerband'][p]:
                df.loc[i, 'in_uptrend'] = False
            else:
                df.loc[i, 'in_uptrend'] = df.loc[p,'in_uptrend']

                if (df['in_uptrend'][i] == True) and df['lowerband'][i] < df['lowerband'][p] :
                    df.loc[i, 'lowerband'] = df.loc[p, 'lowerband']

                if (df['in_uptrend'][i] == False) and df['upperband'][i] > df['upperband'][p]:
                    df.loc[i, 'upperband'] = df.loc[p, 'upperband']

        print(f'\n----------- Analyze supertrend(4h) --------------')
        pprint(df.iloc[-1])

        prev = df.iloc[-2]['in_uptrend']
        curr = df.iloc[-1]['in_uptrend'] 
        sell = (curr == False) and (prev == True)

        global supertrend_sell_decision
        supertrend_sell_decision[symbol] = sell

    except Exception as e:
        print("Exception : ", str(e))

def log_order(symbol, order_type,  price, amount):
    logging.info(f"[ {symbol} ] {order_type} order placed at price = {price}, amount = {amount}")

def log_cancel(symbol, order_type, price):
    logging.info(f"[ {symbol} ] {order_type} cancel for low balance at price= {price}")

def market_buy_coin(exchange, symbol, amount):
    amount_krw = amount 
    exchange.options['createMarketBuyOrderRequiresPrice']=False
    order = exchange.create_market_buy_order(symbol = symbol, amount = amount_krw)
    return order

def market_sell_coin(exchange, symbol, amount, price):
    exchange.options['createMarketBuyOrderRequiresPrice']=False
    sell_amount = round(amount/price, 5)
    order= exchange.create_market_sell_order(symbol=symbol, amount = sell_amount )

def pullback_order(exchange, symbol, price, amount):
    try:
        pb_price = round(random.uniform(0.015, 0.05)*price, 1)
        pb_amount = amount * pullback_portion

        free_KRW = exchange.fetchBalance()['KRW']['free']
        if free_KRW <(bp+amount ):
            log_cancel(symbol, "Pullback buy order", pb_price) 
            return

        order_amount = round(pb_amount/pb_price, 3)

        resp = exchange.create_limit_buy_order(symbol = symbol, amount = order_amount, price = pb_price)
        save_trading_data(symbol,"pullback", "buy", pb_price, order_amount) 
        log_order(symbol, "Pullback buy", pb_price, pb_amount )

    except Exception as e:
        print("Exception : ", str(e))

def mfi_sell_coin(exchange, symbol: str):
    try:

        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['asks'][0][0]
        amount    = calc_mfi_amount(symbol)

        order = market_sell_coin(exchange, symbol, amount, price)
        save_trading_data(symbol,'mfi', 'sell', price, amount) 
        pullback_order(exchange, symbol, price, amount)
        log_order(symbol, "MFI(14), 5m, sell", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        print("Exception : ", str(e))

def cci_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0] 
        amount    = cci_buy_amount

        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW < amount:
            log_cancel_order(symbol, "CCI 5m buy", price)
            return

        market_buy_coin(exchange, symbol, amount, price)
        save_trading_data(symbol,"cci", "buy", price, amount) 
        log_order(symbol, "CCI, 5m, Buy", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0] 
        amount    = stochrsi_buy_amount 

        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW < amount:
            log_cancel_order(symbol, "STOCHRSI buy", price)
            return

        market_buy_coin(exchange, symbol, amount)
        save_trading_data(symbol,"STOCHRSI", "buy", price, amount) 

        logging.info(f"STOCHRSI(10m) buy order placed for {symbol} at price: {price}, amount = {amount}")
        show_orderbook(orderbook)

    except Exception as e:
        print("Exception : ", str(e))

def supertrend_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0]
        amount    = supertrend_sell_amount

        order = market_sell_coin(exchange, symbol, price, amount)
        save_trading_data(symbol,"supertrend", "sell", price, amount) 
        log_order(symbol, "Supertrend sell", price, amount)

        pullback_order(exchange, symbol, price, amount)
        show_orderbook(orderbook)

    except Exception as e:
        print("Exception : ", str(e))

def execute_mfi_sell(exchange, symbol: str)->None:
    sell = mfi_sell_decision[symbol]

    if sell:
       mfi_sell_coin(exchange, symbol)

def execute_cci_buy(exchange, symbol: str)->None:
    buy = cci_buy_decision[symbol]

    if buy:
        cci_buy_coin(exchange, symbol)

def execute_stochrsi_buy(exchange, symbol: str)->None:
    buy = stochrsi_buy_decision[symbol]

    if buy:
        stochrsi_buy_coin(exchange, symbol)

def execute_supertrend_sell(exchange, symbol: str):
    sell = supertrend_sell_decision[symbol]

    if sell:
        supertrend_sell_coin(exchange, symbol)

def monitor_signals(symbols : list[str]):
    print("\n---------------- buy/sell order summary -----------------")

    column_name= ["Symbol", "MFI Sell",  "CCI Buy", "STOCHRSI buy", "Supertrend Sell" ]
    orders = pd.DataFrame(columns = column_name)

    for s in symbols:
        orders.loc[len(orders)] = [s, mfi_sell_decision[s],cci_buy_decision[s], stochrsi_buy_decision[s], \
                                   supertrend_sell_decision[s]]
    pprint(orders)

def monitor_balance(exchange):
    try:
        print("\n---------------- fetch balance result  -----------------")
        balance = exchange.fetchBalance()
        pprint(balance)

    except Exception as e:
        print("Exception : ", str(e))

def init_upbit():
    print('\n-----------------Upbit Exchange Initialization-------------------------')
    print(f'Initialized CCXT with version : {ccxt.__version__}')
    exchange = ccxt.upbit(config={
            'apiKey':key['accessKey'],
            'secret':key['secret'],
            'timeout':15000,
            'enableRateLimit': True,
        }
    )
    return exchange

if __name__=='__main__':

    # Configure logging
    logging.basicConfig(filename="./trading.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    exchange = init_upbit()

    #define doge symbol 
    doge = "DOGE/KRW"

    #defile list of symbols 
    symbols= [doge]

    schedule.every(30).seconds.do(analyze_mfi_signal, exchange, doge)
    schedule.every(30).seconds.do(analyze_cci_signal, exchange, doge)
    schedule.every(30).seconds.do(analyze_stochrsi_signal, exchange, doge)
    schedule.every(30).seconds.do(analyze_supertrend_signal, exchange, doge)

    schedule.every(5).minutes.do(execute_mfi_sell, exchange, doge)
    schedule.every(5).minutes.do(execute_cci_buy,exchange, doge)
    schedule.every(10).minutes.do(execute_stochrsi_buy, exchange, doge)
    schedule.every(1).hours.do(execute_supertrend_sell, exchange, doge)

    # monitoring every 30 seconds
    schedule.every(30).seconds.do(monitor_signals, symbols)
    schedule.every(30).seconds.do(monitor_balance, exchange)

    while True:
        schedule.run_pending()
        time.sleep(0.01)
