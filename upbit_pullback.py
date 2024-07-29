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

mfi_high_threshold = 80.0
current_mfi = defaultdict(float)
mfi_weight  = defaultdict(float)
mfi_sell_amount = 5000000
mfi_sell_decision = defaultdict(bool)

current_cci = defaultdict(float)
cci_low_threshold = -140.0
cci_high_threshold = 140.0
cci_buy_amount   = 40000000
cci_buy_decision = defaultdict(bool)
cci_sell_amount  = 40000000
cci_sell_decision = defaultdict(bool)
cci_scalping_amount   = 1000000
cci_scalping_buy_decision = defaultdict(bool)
cci_scalping_sell_decision = defaultdict(bool)

stochrsi_low_threshold = 25.0
stochrsi_high_threshold = 83.0
stochrsi_buy_amount  = 4000000
stochrsi_buy_decision = defaultdict(bool)

is_uptrend = defaultdict(bool)
supertrend_sell_amount = 4000000
supertrend_sell_decision = defaultdict(bool)
supertrend_buy_amount  = 4000000
supertrend_buy_decision = defaultdict(bool)

my_balance = defaultdict(float)

pullback_portion = 0.7

daily_pct_map = defaultdict(lambda: defaultdict(float))
daily_down_state = defaultdict(lambda: defaultdict(bool))
daily_up_state = defaultdict(lambda: defaultdict(bool))

def write_to_csv(row_dict, file_name):

    file_path = file_name
    column_names = ['datetime', 'symbol', 'indicator', 'order_type', 'price', 'amount']
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)

        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()

        writer.writerow(row_dict)

def write_to_volume_csv(row_dict, file_name):

    file_path = file_name
    column_names = ['datetime', 'volume']
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=column_names)

        if not file_exists or os.path.getsize(file_path) == 0:
            writer.writeheader()

        writer.writerow(row_dict)


def save_data(symbol, indicator, order_type, price, amount):

    KST = timezone('Asia/Seoul')

    now = datetime.now().astimezone(KST)
    now_sec = now.strftime("%Y-%m-%d %H:%M:%S")
    csv_row = {
             'datetime': now_sec,
             'symbol': symbol,
             'indicator': indicator,
             'order_type': order_type,
             'price' : round(price, 1),
             'amount': round(amount, 0),
    }

    trading_log_file = 'trading.csv'
    write_to_csv( csv_row , trading_log_file)

def save_volume(currency, volume):

    KST = timezone('Asia/Seoul')
    now = datetime.now().astimezone(KST)
    now_sec = now.strftime("%Y-%m-%d %H:%M:%S")

    volume_row = {
        'datetime': now_sec,
        'volume': round(float(volume), 3),
    }

    volume_log_file = f'volume_{currency}.csv'
    write_to_volume_csv( volume_row , volume_log_file)

def show_orderbook(orderbook):
    print("\n------------Getting order book -----------")
    pprint(orderbook)

def analyze_stochrsi_signal(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='30m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        stochrsi = ta.stochrsi(df['close'], length=14, fastk_period=3, fastd_period=3, append=True)

        df['stochrsi_k'] = stochrsi['STOCHRSIk_14_14_3_3']
        df['stochrsi_d'] = stochrsi['STOCHRSId_14_14_3_3']

        # Get the latest value
        stochrsi_k = df['stochrsi_k'].iloc[-1]
        stochrsi_d = df['stochrsi_d'].iloc[-1]

        # Stoch rsi cross-over strategy
        buy  = (stochrsi_k > stochrsi_d) and (stochrsi_k < stochrsi_low_threshold)

        global stochrsi_buy_decision
        stochrsi_buy_decision[symbol] = buy

        df['stochrsi_buy']  = buy

        print(f'\n----------- {symbol} STOCHRSI Signal Analysis (30 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        logging.info("Exception in analyze_stochrsi_signal: ", str(e))

def analyze_mfi_signal(exchange, symbol: str)->None:
    try:
        ohlcv_5m = exchange.fetch_ohlcv(symbol, timeframe='5m')
        df = pd.DataFrame(ohlcv_5m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi_5m'] = round(ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14), 1)

        mfi_5m = df['mfi_5m'].iloc[-1]

        ohlcv_30m = exchange.fetch_ohlcv(symbol, timeframe='30m')
        df_30m = pd.DataFrame(ohlcv_30m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_30m['datetime'] = pd.to_datetime(df_30m['datetime'], utc=True, unit='ms')
        df_30m['datetime'] = df_30m['datetime'].dt.tz_convert("Asia/Seoul")
        df_30m['mfi_30m'] = round(ta.mfi(df_30m['high'], df_30m['low'], df_30m['close'], df_30m['volume'], length=14), 1)

        mfi_30m = df_30m['mfi_30m'].iloc[-1]

        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h')
        df_1h = pd.DataFrame(ohlcv_1h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['datetime'] = pd.to_datetime(df_1h['datetime'], utc=True, unit='ms')
        df_1h['datetime'] = df_1h['datetime'].dt.tz_convert("Asia/Seoul")
        df_1h['mfi_1h'] = round(ta.mfi(df_1h['high'], df_1h['low'], df_1h['close'], df_1h['volume'], length=14), 1)

        mfi_1h = df_1h['mfi_1h'].iloc[-1]

        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df_4h = pd.DataFrame(ohlcv_4h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], utc=True, unit='ms')
        df_4h['datetime'] = df_4h['datetime'].dt.tz_convert("Asia/Seoul")
        df_4h['mfi_4h'] = round(ta.mfi(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume'], length=14), 1)

        mfi_4h = df_4h['mfi_4h'].iloc[-1]

        ohlcv_1d = exchange.fetch_ohlcv(symbol, timeframe='1d')
        df_1d = pd.DataFrame(ohlcv_1d, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1d['datetime'] = pd.to_datetime(df_1d['datetime'], utc=True, unit='ms')
        df_1d['datetime'] = df_1d['datetime'].dt.tz_convert("Asia/Seoul")
        df_1d['mfi_1d'] = round(ta.mfi(df_1d['high'], df_1d['low'], df_1d['close'], df_1d['volume'], length=14), 1)
        df_1d['mfi_14d_ma'] = df_1d['mfi_1d'].rolling(window=14).mean()

        mfi_1d = df_1d['mfi_1d'].iloc[-1]
        mfi_14d = df_1d['mfi_14d_ma'].iloc[-1]

        print(f'\n----------- {symbol} MFI Signal Analysis ( 5 minutes) --------------')
        pprint(df.iloc[-1])
        print(f'\n----------- {symbol} MFI Signal Analysis (30 minutes) --------------')
        pprint(df_30m.iloc[-1])
        print(f'\n----------- {symbol} MFI Signal Analysis ( 1 hour) --------------')
        pprint(df_1h.iloc[-1])
        print(f'\n----------- {symbol} MFI Signal Analysis ( 4 hour) --------------')
        pprint(df_4h.iloc[-1])
        print(f'\n----------- {symbol} MFI Signal Analysis ( 1 day and 7 day) --------------')
        pprint(df_1d.iloc[-1])

        mfi = (mfi_5m + mfi_30m + mfi_1h + mfi_4h + mfi_1d + mfi_14d)/6.0

        global mfi_sell_decision
        mfi_sell_decision[symbol] = (mfi*current_cci[symbol]/140.0) > mfi_high_threshold

        global current_mfi
        current_mfi[symbol] = mfi

        global mfi_weight
        mfi_weight[symbol] = round(abs(mfi - 50)/20.0, 1)

    except Exception as e:
        logging.info("Exception in analyze_mfi_signal ", str(e))

def analyze_cci_scalping_signal(exchange, symbol: str)->None:
    try:
        ohlcv_3m = exchange.fetch_ohlcv(symbol, timeframe='3m')
        df = pd.DataFrame(ohlcv_3m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['cci_3m']   = round(ta.cci(df['high'], df['low'], df['close'], length=14), 1)

        cci_3m = df['cci_3m'].iloc[-1]

        ohlcv_30m = exchange.fetch_ohlcv(symbol, timeframe='30m')
        df_30m = pd.DataFrame(ohlcv_30m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_30m['datetime'] = pd.to_datetime(df_30m['datetime'], utc=True, unit='ms')
        df_30m['datetime'] = df_30m['datetime'].dt.tz_convert("Asia/Seoul")
        df_30m['cci_30m']   = round(ta.cci(df_30m['high'], df_30m['low'], df_30m['close'], length=14), 1)

        cci_30m= df_30m['cci_30m'].iloc[-1]

        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df_4h = pd.DataFrame(ohlcv_4h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], utc=True, unit='ms')
        df_4h['datetime'] = df_4h['datetime'].dt.tz_convert("Asia/Seoul")
        df_4h['cci_4h']   = round(ta.cci(df_4h['high'], df_4h['low'], df_4h['close'], length=14), 1)

        cci_4h = df_4h['cci_4h'].iloc[-1]

        cci = (cci_3m + cci_30m*0.8)/2.0

        buy  = (cci < -130) and (cci_4h < -70)
        sell = (cci > 130) and (cci_4h > 70)

        global cci_scalping_buy_decision
        global cci_scalping_sell_decision

        cci_scalping_buy_decision[symbol] = buy
        cci_scalping_sell_decision[symbol] = sell

        print(f'\n----------- {symbol} CCI Scalping Signal Analysis ( 3m and 30m average) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        logging.info("Exception in analyze_cci_signal : %s", str(e))


def analyze_daily_pct(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")


        # Calculate consecutive day percentage change up to 10 days
        for i in range(1, 11):
            df[f'{i}d_pct'] = df['close'].pct_change(periods=i) * 100
        print(f'\n-----------{symbol} last 10 day consecutive --------------')
        x = df.iloc[-1].drop("volume")
        pprint(x)

        global daily_pct_map
        daily_pct_map[symbol] = x

        three_day_down = (x['1d_pct'] < -0.01 ) and (x['3d_pct'] < x['1d_pct'])
        five_day_down = three_day_down and (x['5d_pct'] < x['3d_pct'])
        seven_day_down = five_day_down and (x['7d_pct'] < x['5d_pct'])
        nine_day_down = seven_day_down and (x['9d_pct'] < x['7d_pct'])

        three_day_up = (x['1d_pct'] > 0.01 ) and (x['3d_pct'] > x['1d_pct'])
        five_day_up = three_day_up and (x['5d_pct'] > x['3d_pct'])
        seven_day_up = five_day_up and (x['7d_pct'] > x['5d_pct'])
        nine_day_up = seven_day_up and (x['9d_pct'] > x['7d_pct']) 

        global daily_down_state
        daily_down_state[symbol]['3d_down'] = three_day_down
        daily_down_state[symbol]['5d_down'] = five_day_down
        daily_down_state[symbol]['7d_down'] = seven_day_down
        daily_down_state[symbol]['9d_down'] = nine_day_down

        global daily_up_state
        daily_up_state[symbol]['3d_up'] = three_day_up
        daily_up_state[symbol]['5d_up'] = five_day_up
        daily_up_state[symbol]['7d_up'] = seven_day_up
        daily_up_state[symbol]['9d_up'] = nine_day_up

        print(f'\n-----------{symbol} consecutive down state --------------')
        pprint(daily_down_state[symbol])

        print(f'\n-----------{symbol} consecutive up state --------------')
        pprint(daily_up_state[symbol])

    except Exception as e:
        logging.info("Exception in analyze_daily_pct : %s", str(e))


def analyze_cci_signal(exchange, symbol: str)->None:
    try:
        ohlcv_5m = exchange.fetch_ohlcv(symbol, timeframe='5m')
        df = pd.DataFrame(ohlcv_5m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['cci_5m']   = round(ta.cci(df['high'], df['low'], df['close'], length=14), 1)

        cci_5m = df['cci_5m'].iloc[-1]

        ohlcv_30m = exchange.fetch_ohlcv(symbol, timeframe='30m')
        df_30m = pd.DataFrame(ohlcv_30m, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_30m['datetime'] = pd.to_datetime(df_30m['datetime'], utc=True, unit='ms')
        df_30m['datetime'] = df_30m['datetime'].dt.tz_convert("Asia/Seoul")
        df_30m['cci_30m']   = round(ta.cci(df_30m['high'], df_30m['low'], df_30m['close'], length=14), 1)

        cci_30m= df_30m['cci_30m'].iloc[-1]

        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe='1h')
        df_1h = pd.DataFrame(ohlcv_1h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1h['datetime'] = pd.to_datetime(df_1h['datetime'], utc=True, unit='ms')
        df_1h['datetime'] = df_1h['datetime'].dt.tz_convert("Asia/Seoul")
        df_1h['cci_1h']   = round(ta.cci(df_1h['high'], df_1h['low'], df_1h['close'], length=14), 1)

        cci_1h = df_1h['cci_1h'].iloc[-1]

        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df_4h = pd.DataFrame(ohlcv_4h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], utc=True, unit='ms')
        df_4h['datetime'] = df_4h['datetime'].dt.tz_convert("Asia/Seoul")
        df_4h['cci_4h']   = round(ta.cci(df_4h['high'], df_4h['low'], df_4h['close'], length=14), 1)

        cci_4h = df_4h['cci_4h'].iloc[-1]

        ohlcv_1d = exchange.fetch_ohlcv(symbol, timeframe='1d')
        df_1d = pd.DataFrame(ohlcv_1d, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1d['datetime'] = pd.to_datetime(df_1d['datetime'], utc=True, unit='ms')
        df_1d['datetime'] = df_1d['datetime'].dt.tz_convert("Asia/Seoul")
        df_1d['cci_1d']   = round(ta.cci(df_1d['high'], df_1d['low'], df_1d['close'], length=14), 1)
        df_1d['cci_14d_ma'] = df_1d['cci_1d'].rolling(window=14).mean()

        cci_1d = df_1d['cci_1d'].iloc[-1]
        cci_14d = df_1d['cci_14d_ma'].iloc[-1]

        cci = (cci_5m + cci_30m + cci_1h + cci_4h + cci_1d + cci_14d)/6.0

        global current_cci
        current_cci[symbol] = cci

        global cci_sell_decision
        cci_sell_decision[symbol] = (cci > cci_high_threshold)

        global cci_buy_decision
        cci_buy_decision[symbol] = (cci < cci_low_threshold )

        print(f'\n----------- {symbol} CCI Signal Analysis ( 5 minutes) --------------')
        pprint(df.iloc[-1])
        print(f'\n----------- {symbol} CCI Signal Analysis (30 minutes) --------------')
        pprint(df_30m.iloc[-1])
        print(f'\n----------- {symbol} CCI Signal Analysis ( 1 hour ) --------------')
        pprint(df_1h.iloc[-1])
        print(f'\n----------- {symbol} CCI Signal Analysis ( 4 hour ) --------------')
        pprint(df_4h.iloc[-1])
        print(f'\n----------- {symbol} CCI Signal Analysis ( 1 day and 7 day ) --------------')
        pprint(df_1d.iloc[-1])

    except Exception as e:
        logging.info("Exception in analyze_cci_signal : ", str(e))

def analyze_supertrend_signal(exchange, symbol: str)->None:
    try:
        # upto two weeks analyze supertrend 
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        period = 14
        multiplier = 2.0

        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)

        df['upperband'] = (df['high'] + df['low'])/2 + (multiplier * df['atr'])
        df['lowerband'] = (df['high'] + df['low'])/2 - (multiplier * df['atr'])

        df['uptrend'] = False

        for i in range(1, len(df.index)):
            p = i - 1

            if df['close'][i] > df['upperband'][p]:
                df.loc[i, 'uptrend'] = True
            elif df['close'][i] < df['lowerband'][p]:
                df.loc[i, 'uptrend'] = False
            else:
                df.loc[i, 'uptrend'] = df.loc[p,'uptrend']

                if df['uptrend'][i] and (df['lowerband'][i] < df['lowerband'][p]) :
                    df.loc[i, 'lowerband'] = df.loc[p, 'lowerband']

                if (not df['uptrend'][i] ) and (df['upperband'][i] > df['upperband'][p]):
                    df.loc[i, 'upperband'] = df.loc[p, 'upperband']

        print('\n----------- Analyze supertrend(1 hour) --------------')
        pprint(df.iloc[-1])

        uptrend = df.iloc[-1]['uptrend']

        global is_uptrend
        is_uptrend[symbol] = uptrend

        global supertrend_sell_decision
        supertrend_sell_decision[symbol] = uptrend and (current_cci[symbol] > 100)

        global supertrend_buy_decision
        supertrend_buy_decision[symbol] =  (not uptrend) and (current_cci[symbol] < -100)

    except Exception as e:
        logging.info("Exception in analyze_supertrend_signal: ", str(e))

def log_order(symbol, order_type,  price, amount):
    logging.info(f"[ {symbol} ] {order_type} order placed at price = {price}, amount = {amount}")

def log_cancel(symbol, order_type, price):
    logging.info(f"[ {symbol} ] {order_type} cancel for low balance at price= {price}")

def market_buy_coin(exchange, symbol, amount):
    amount_krw = amount
    exchange.options['createMarketBuyOrderRequiresPrice']=False
    exchange.create_market_buy_order(symbol = symbol, amount = amount_krw)

def market_sell_coin(exchange, symbol, amount, price):
    exchange.options['createMarketBuyOrderRequiresPrice']=False
    sell_amount = round(amount/price, 3)

    if my_balance[symbol] >= sell_amount:
        exchange.create_market_sell_order(symbol=symbol, amount = sell_amount )

def calc_pullback_price(symbol, price) -> float:
    r = abs(random.gauss(0.035, 0.015))
    return round(price * (1 - r), 1)

def pullback_order(exchange, symbol, price, amount):
    try:
        pb_price = calc_pullback_price(symbol, price)
        pb_amount = amount * pullback_portion

        free_KRW = exchange.fetchBalance()['KRW']['free']
        if free_KRW < pb_amount :
            return

        order_amount = round(pb_amount/pb_price, 2)
        exchange.create_limit_buy_order(symbol = symbol, amount = order_amount, price = pb_price)
        save_data(symbol,"pullback", "buy", pb_price, order_amount)
        log_order(symbol, "Pullback buy", pb_price, pb_amount )

    except Exception as e:
        logging.info("Exception : ", str(e))

def scalping_pullback_order(exchange, symbol, price, amount):
    try:
        pb_price = calc_pullback_price(symbol, price)
        pb_amount = amount * 0.6

        free_KRW = exchange.fetchBalance()['KRW']['free']
        if free_KRW < pb_amount :
            return

        order_amount = round(pb_amount/pb_price, 2)
        exchange.create_limit_buy_order(symbol = symbol, amount = order_amount, price = pb_price)
        save_data(symbol,"pullback", "buy", pb_price, order_amount)
        log_order(symbol, "Pullback buy", pb_price, pb_amount )

    except Exception as e:
        logging.info("Exception : ", str(e))


def calc_mfi_amount(symbol):
    amount = mfi_sell_amount * mfi_weight[symbol]
    return amount

def mfi_sell_coin(exchange, symbol: str):
    try:

        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['asks'][0][0]
        amount    = calc_mfi_amount(symbol)

        market_sell_coin(exchange, symbol, amount, price)
        save_data(symbol,'mfi', 'sell', price, amount) 
        pullback_order(exchange, symbol, price, amount)
        log_order(symbol, "MFI(14), 5m, sell", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def cci_scalping_sell_coin(exchange, symbol: str):
    try:

        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['asks'][0][0]
        amount    = cci_scalping_amount

        market_sell_coin(exchange, symbol, amount, price)
        scalping_pullback_order(exchange, symbol, price, amount)
        save_data(symbol,'CCI scalping', 'sell', price, amount) 
        log_order(symbol, "CCI scalping sell", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def cci_scalping_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0] 
        amount    = cci_scalping_amount

        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW < amount:
            return

        market_buy_coin(exchange, symbol, amount)
        save_data(symbol,"cci scalping", "buy", price, amount) 
        log_order(symbol, "CCI scalping, 3m, Buy", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def cci_sell_coin(exchange, symbol: str):
    try:

        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['asks'][0][0]
        amount    = cci_sell_amount 

        market_sell_coin(exchange, symbol, amount, price)
        save_data(symbol,'CCI', 'sell', price, amount) 
        pullback_order(exchange, symbol, price, amount)
        log_order(symbol, "CCI average based sell", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))


def cci_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0] 
        amount    = cci_buy_amount

        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW < amount:
            return

        market_buy_coin(exchange, symbol, amount)
        save_data(symbol,"CCI", "buy", price, amount) 
        log_order(symbol, "CCI, 5m, Buy", price, amount)

        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def stochrsi_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0] 
        amount    = stochrsi_buy_amount 

        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW < amount:
            return

        market_buy_coin(exchange, symbol, amount)
        save_data(symbol,"STOCHRSI", "buy", price, amount) 

        logging.info(f"STOCHRSI(10m) buy order placed for {symbol} at price: {price}, amount = {amount}")
        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def supertrend_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0]
        amount    = supertrend_sell_amount

        market_sell_coin(exchange, symbol, price, amount)
        save_data(symbol,"supertrend", "sell", price, amount) 
        log_order(symbol, "Supertrend sell", price, amount)

        pullback_order(exchange, symbol, price, amount)
        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def supertrend_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = orderbook['bids'][0][0]
        amount    = supertrend_buy_amount

        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW < amount:
            return

        market_buy_coin(exchange, symbol, amount)
        save_data(symbol,"Supertrend", "buy", price, amount)

        logging.info(f"Supertrend buy order placed for {symbol} at price: {price}, amount = {amount}")
        show_orderbook(orderbook)

    except Exception as e:
        logging.info("Exception : ", str(e))

def execute_mfi_sell(exchange, symbol: str)->None:
    sell = mfi_sell_decision[symbol]

    if sell:
       mfi_sell_coin(exchange, symbol)

def execute_cci_scalping_sell(exchange, symbol: str)->None:
    sell = cci_scalping_sell_decision[symbol]

    if sell:
        cci_scalping_sell_coin(exchange, symbol)

def execute_cci_scalping_buy(exchange, symbol: str)->None:
    buy = cci_scalping_buy_decision[symbol]

    if buy:
        cci_scalping_buy_coin(exchange, symbol)

def execute_cci_sell(exchange, symbol: str)->None:
    sell = cci_sell_decision[symbol]

    if sell:
        cci_sell_coin(exchange, symbol)

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

def execute_supertrend_buy(exchange, symbol: str):
    sell = supertrend_buy_decision[symbol]

    if sell:
        supertrend_buy_coin(exchange, symbol)

def monitor_signals(symbols : list[str]):
    print("\n---------------- buy/sell order summary -----------------")

    column_name= ["Symbol", "MFI Sell", "CCI Sell", "CCI Buy", "STOCHRSI buy", "SuperTrend Sell", "SuperTrend Buy"]
    orders = pd.DataFrame(columns = column_name)

    for s in symbols:
        orders.loc[len(orders)] = [s, mfi_sell_decision[s],cci_sell_decision[s], cci_buy_decision[s], stochrsi_buy_decision[s], \
                                   supertrend_sell_decision[s], supertrend_buy_decision[s]]
    pprint(orders)

    ta_index_name = ["Symbol", "Current CCI", "Current MFI"]
    cci_mfi_analysis = pd.DataFrame(columns = ta_index_name)
    for s in symbols:
        cci_mfi_analysis.loc[len(cci_mfi_analysis)]= [s, current_cci[s], current_mfi[s]]

    print("\n---------------- average cci and mfi values  -----------------")
    pprint(cci_mfi_analysis)

def extract_balances(balance):
    return balance.get('total', {})

def monitor_balance(exchange):
    try:
        print("\n---------------- fetch balance result  -----------------")
        balance = exchange.fetchBalance()
        pprint(balance)
        print("\n---------------- extract balance result  -----------------")
        info_balances = extract_balances(balance) 
        pprint(info_balances)

    except Exception as e:
        print("Exception : ", str(e))

def monitor_volume(exchange):
    try:
        balance = exchange.fetchBalance()
        info_balances = extract_balances(balance)

        print("\n---------------- monitor volume  -----------------")
        for currency, amount in info_balances.items():
            if currency != 'KRW':
                symbol = currency +"/KRW"
                global my_balance
                my_balance[symbol] = amount
                pprint( f'currency = {currency}, amount = {amount}, symbol={symbol}, my_balance[{symbol}]={my_balance[symbol]}')
                save_volume(currency, amount)

    except Exception as e:
        print("Exception : ", str(e))

def monitor_daily_pct():
    print("\n---------------- monitor daily pct  -----------------")
    pprint(daily_pct_map)


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
    btc = "BTC/KRW"
    eth = "ETH/KRW"
    sol = "SOL/KRW"

    #defile list of symbols 
    symbols= [doge, sol, eth, btc]

    schedule.every(10).seconds.do(analyze_cci_scalping_signal, exchange, doge)
    schedule.every(10).seconds.do(analyze_daily_pct, exchange, doge)
    schedule.every(10).seconds.do(analyze_mfi_signal, exchange, doge)
    schedule.every(10).seconds.do(analyze_cci_signal, exchange, doge)
    schedule.every(10).seconds.do(analyze_stochrsi_signal, exchange, doge)
    schedule.every(10).seconds.do(analyze_supertrend_signal, exchange, doge)

    schedule.every(1).minutes.do(execute_cci_scalping_buy, exchange, doge)
    schedule.every(1).minutes.do(execute_cci_scalping_sell, exchange, doge)
    schedule.every(5).minutes.do(execute_mfi_sell, exchange, doge)
    schedule.every(5).minutes.do(execute_cci_buy, exchange, doge)
    schedule.every(5).minutes.do(execute_cci_sell, exchange, doge)
    schedule.every(30).minutes.do(execute_stochrsi_buy, exchange, doge)
    schedule.every(30).minutes.do(execute_supertrend_sell, exchange, doge)
    schedule.every(30).minutes.do(execute_supertrend_buy, exchange, doge)

    schedule.every(10).seconds.do(analyze_cci_scalping_signal, exchange, btc)
    schedule.every(10).seconds.do(analyze_daily_pct, exchange, btc)
    schedule.every(10).seconds.do(analyze_mfi_signal, exchange, btc)
    schedule.every(10).seconds.do(analyze_cci_signal, exchange, btc)
    schedule.every(10).seconds.do(analyze_stochrsi_signal, exchange, btc)
    schedule.every(10).seconds.do(analyze_supertrend_signal, exchange, btc)

    schedule.every(3).minutes.do(execute_cci_scalping_buy, exchange, btc)
    schedule.every(3).minutes.do(execute_cci_scalping_sell, exchange, btc)
    schedule.every(5).minutes.do(execute_mfi_sell, exchange, btc)
    schedule.every(5).minutes.do(execute_cci_buy, exchange, btc)
    schedule.every(5).minutes.do(execute_cci_sell, exchange, btc)
    schedule.every(30).minutes.do(execute_stochrsi_buy, exchange, btc)
    schedule.every(30).minutes.do(execute_supertrend_sell, exchange, btc)
    schedule.every(30).minutes.do(execute_supertrend_buy, exchange, btc)

    schedule.every(10).seconds.do(analyze_cci_scalping_signal, exchange, sol)
    schedule.every(10).seconds.do(analyze_daily_pct, exchange, sol)
    schedule.every(10).seconds.do(analyze_mfi_signal, exchange, sol)
    schedule.every(10).seconds.do(analyze_cci_signal, exchange, sol)
    schedule.every(10).seconds.do(analyze_stochrsi_signal, exchange, sol)
    schedule.every(10).seconds.do(analyze_supertrend_signal, exchange, sol)

    schedule.every(3).minutes.do(execute_cci_scalping_buy, exchange, sol)
    schedule.every(3).minutes.do(execute_cci_scalping_sell, exchange, sol)
    schedule.every(5).minutes.do(execute_mfi_sell, exchange, sol)
    schedule.every(5).minutes.do(execute_cci_buy, exchange, sol)
    schedule.every(5).minutes.do(execute_cci_sell, exchange, sol)
    schedule.every(30).minutes.do(execute_stochrsi_buy, exchange, sol)
    schedule.every(30).minutes.do(execute_supertrend_sell, exchange, sol)
    schedule.every(30).minutes.do(execute_supertrend_buy, exchange, sol)

    schedule.every(10).seconds.do(analyze_cci_scalping_signal, exchange, eth)
    schedule.every(10).seconds.do(analyze_daily_pct, exchange, eth)
    schedule.every(10).seconds.do(analyze_mfi_signal, exchange, eth)
    schedule.every(10).seconds.do(analyze_cci_signal, exchange, eth)
    schedule.every(10).seconds.do(analyze_stochrsi_signal, exchange, eth)
    schedule.every(10).seconds.do(analyze_supertrend_signal, exchange, eth)

    schedule.every(3).minutes.do(execute_cci_scalping_buy, exchange, eth)
    schedule.every(3).minutes.do(execute_cci_scalping_sell, exchange, eth)
    schedule.every(5).minutes.do(execute_mfi_sell, exchange, eth)
    schedule.every(5).minutes.do(execute_cci_buy, exchange, eth)
    schedule.every(5).minutes.do(execute_cci_sell, exchange, eth)
    schedule.every(30).minutes.do(execute_stochrsi_buy, exchange, eth)
    schedule.every(30).minutes.do(execute_supertrend_sell, exchange, eth)
    schedule.every(30).minutes.do(execute_supertrend_buy, exchange, eth)

    # monitoring every 30 seconds
    schedule.every(30).seconds.do(monitor_signals, symbols)
    schedule.every(30).seconds.do(monitor_balance, exchange)
    schedule.every(30).seconds.do(monitor_volume, exchange)
    schedule.every(30).seconds.do(monitor_daily_pct)

    while True:
        schedule.run_pending()
        time.sleep(0.005)
