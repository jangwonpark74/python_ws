import ccxt
import talib
import time
import schedule
import pandas as pd
import logging
import time
import pandas_ta as ta

from pprint import pprint
from collections import defaultdict

from conf import key
from conf import binance_key

# Bollinger band buy/sell order 
bollinger_buy = defaultdict(bool)
bollinger_sell = defaultdict(bool)

# Bollinger threshold for logging
bollinger_threshold = defaultdict(float)
bollinger_width = defaultdict(float)

# Scalping every 3 minutes based on RSI(3m) indicator
rsi_3m_scalping_buy = defaultdict(bool)
rsi_3m_scalping_sell= defaultdict(bool)

# scalping every 5 minute based on MFI(5m) indicator 
mfi_5m_scalping_buy = defaultdict(bool)
mfi_5m_scalping_sell= defaultdict(bool)

# scalping every 1 hour based on MFI(4h) indicator 
mfi_4h_scalping_sell = defaultdict(bool)
mfi_4h_scalping_buy = defaultdict(bool)

# Bollinger band analysis based buy, sell amount
bb_trading_amount = 2000000

# RSI 3 minute scalping amount 
rsi_3m_scalping_sell_amount = 3000000
rsi_3m_scalping_buy_amount  = 3000000

# MFI 5 minute scalping amount 
mfi_5m_scalping_sell_amount = 2000000
mfi_5m_scalping_buy_amount  = 2000000

# MFI 4 hour scalping amount 
mfi_4h_scalping_sell_amount = 2000000
mfi_4h_scalping_buy_amount  = 2000000

# STOCHRSI 3 minutes amount 
stochrsi_5m_sell_amount = 3000000
stochrsi_5m_buy_amount  = 3000000

# 4 Hour stochrsi amount 
stochrsi_4h_sell_amount = 4000000
stochrsi_4h_buy_amount  = 4000000

# MFI 4 hour for volatility analysis
mfi_4h = defaultdict(float)

# RSI 3m high low threshold
rsi_3m_high_threshold = 70
rsi_3m_low_threshold = 25
rsi_low_threshold = 25

# MFI high low threshold
mfi_high_threshold = 83
mfi_low_threshold = 25

# Define parameters for Stochastic RSI
overbought_threshold = 80
oversold_threshold = 25

# StochRSI(5m) sell buy every 5 minutes
stochrsi_5m_sell = defaultdict(bool)
stochrsi_5m_buy = defaultdict(bool)

# StochRSI (4h) sell buy every 1 hour 
stochrsi_4h_sell = defaultdict(bool)
stochrsi_4h_buy = defaultdict(bool)

# Global variable to keep the count for max 15 minutes continue for order
iterations = defaultdict(int)

# Global variable to keep supertrend sell count
supertrend_sell_iter = defaultdict(int)

# Current supertrend 
supertrend_up = defaultdict(bool)

# Supertrend buy, sell
supertrend_buy = defaultdict(bool)
supertrend_sell = defaultdict(bool)

# supertrend buy amount at every 2 hour
supertrend_buy_amount = 2000000

# supertrend sell one time 
supertrend_sell_quota = defaultdict(float) 
supertrend_sell_amount = defaultdict(float)

pd.set_option('display.max_rows', None)

def reset_bollinger_order(symbol: str):
    global bollinger_sell
    global bollinger_buy
    bollinger_sell[symbol] = False
    bollinger_buy[symbol] = False

def calc_volatility(x: float) -> float:
    volatility = round(-0.0012 * x * x + 0.12 * x, 2)
    return volatility

def analyze_historical_data(exchange, symbol:str):
    try:
        from_ts = exchange.parse8601('2021-01-01 00:00:00')
        ohlcv_list = []
        ohlcv = exchange.fetch_ohlcv(symbol, '15m', since=from_ts, limit=1000)
        ohlcv_list.append(ohlcv)
        while True:
            from_ts = ohlcv[-1][0]
            new_ohlcv = exchange.fetch_ohlcv(symbol,'15m', since=from_ts, limit=1000)
            if len(new_ohlcv) != 1000:
                break
    except Exception as e:
        print("Exception : ", str(e))

def analyze_signals_1d(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        df['rsi']      = talib.RSI(df['close'], timeperiod=14)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'])
        df['bollinger_width'] = round(((df['bollinger_upper'] - df['bollinger_lower'])/df['bollinger_middle']) * 100, 1)
        df['bollinger_upper'] = round(df['bollinger_upper'], 1)
        df['bollinger_middle']= round(df['bollinger_middle'], 1)
        df['bollinger_lower'] = round(df['bollinger_lower'], 1)

        print(f'\n----------------------- {symbol} Signal Analysis ( 1 day ) -----------------------------')
        pprint(df.iloc[-1])

        # daily mfi update 
        mfi = df['mfi'].iloc[-1]

    except Exception as e:
        print("Exception : ", str(e))

def analyze_signals_4h(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        df['rsi']      = talib.RSI(df['close'], timeperiod=14)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'])

        # Scalping based on MFI and RSI every 4 hours
        mfi = df['mfi'].iloc[-1]
        rsi = df['rsi'].iloc[-1]

        sell = mfi > mfi_high_threshold
        buy  = (mfi < mfi_low_threshold) | (rsi < rsi_low_threshold)

        # update data for execution of order
        global mfi_4h_scalping_sell
        global mfi_4h_scalping_buy
        mfi_4h_scalping_sell[symbol] = sell
        mfi_4h_scalping_buy[symbol] = buy

        # store information for dispaly
        df['mfi_4h_scalping_sell'] = sell
        df['mfi_4h_buy']  = buy

        # update global variable mfi_4h for volatility calculation
        global mfi_4h
        mfi_4h[symbol] = mfi

        print(f'\n----------- {symbol} MFI analysis (4 hour) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))


def analyze_bb_signals_15m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = round( talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 1)
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'])
        df['bollinger_upper']  = round(df['bollinger_upper'], 1)
        df['bollinger_middle'] = round(df['bollinger_middle'], 1)
        df['bollinger_lower']  = round(df['bollinger_lower'], 1)
        df['bollinger_width']  = round(((df['bollinger_upper'] - df['bollinger_lower'])/df['bollinger_middle']) * 100 , 3)

        # bollinger volatility based sell buy decision with bollinger width threshold
        bb_width = df['bollinger_width'].iloc[-1]

        # sell, buy condition check
        mfi = mfi_4h[symbol] 
        threshold = calc_volatility(mfi)
        sell = (df['high'].iloc[-1] > df['bollinger_upper'].iloc[-1]) and (bb_width > threshold) 
        buy  = (df['low'].iloc[-1] < df['bollinger_lower'].iloc[-1]) and  (bb_width > threshold)

        df['bollinger_sell'] = sell
        df['bollinger_buy'] = buy

        if sell | buy :
            bollinger_threshold[symbol] = threshold
            bollinger_width[symbol] = bb_width
        else:
            bollinger_threshold[symbol] = 0.0
            bollinger_width[symbol] = 0.0

        global bollinger_sell
        global bollinger_buy

        bollinger_sell[symbol] = sell
        bollinger_buy[symbol] = buy

        print(f'\n----------- {symbol} Bollinger Sell/Buy and Volatiltiy Analysis (15 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_stochrsi_5m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m')
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
        sell = current_stochrsi_k < current_stochrsi_d and current_stochrsi_k > overbought_threshold
        buy = current_stochrsi_k > current_stochrsi_d and current_stochrsi_k < oversold_threshold 

        df['stochrsi_sell'] = sell
        df['stochrsi_buy'] = buy

        # update data for execution of order
        global stochrsi_5m_sell
        global stochrsi_5m_buy
        stochrsi_5m_sell[symbol] = sell
        stochrsi_5m_buy[symbol] = buy

        print(f'\n----------- {symbol} STOCHRSI Signal Analysis (5 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_mfi_signals_5m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = round(talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 2)
        df['rsi']      = round(talib.RSI(df['close'], timeperiod=14), 2)

        # Scalping based on 5 minutes MFI  
        mfi = df['mfi'].iloc[-1]
        rsi = df['rsi'].iloc[-1]

        sell = mfi > mfi_high_threshold
        buy  = (mfi < mfi_low_threshold) | (rsi < rsi_low_threshold)

        # update data for execution of order
        global mfi_5m_scalping_sell
        global mfi_5m_scalping_buy
        mfi_5m_scalping_sell[symbol] = sell
        mfi_5m_scalping_buy[symbol] = buy

        # store information for dispaly
        df['mfi_5m_scalping_sell'] = sell
        df['mfi_5m_scalping_buy']  = buy

        print(f'\n----------- {symbol} Signal Analysis (10 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_rsi_signals_3m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='3m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        df['rsi'] = ta.rsi(df['close'], length=14)

        # Get the latest value
        rsi_3m = df['rsi'].iloc[-1]

        # Stoch rsi cross-over strategy
        sell = rsi_3m > rsi_3m_high_threshold
        buy = rsi_3m < rsi_3m_low_threshold

        df['rsi_3m_sell'] = sell
        df['rsi_3m_buy'] = buy

        # update data for execution of order
        global rsi_3m_sell
        global rsi_3m_buy
        rsi_3m_sell[symbol] = sell
        rsi_3m_buy[symbol] = buy

        print(f'\n----------- {symbol} RSI Signal Analysis (3 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_stochrsi_4h(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h')
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
        sell = current_stochrsi_k < current_stochrsi_d and current_stochrsi_k > overbought_threshold
        buy = current_stochrsi_k > current_stochrsi_d and current_stochrsi_k < oversold_threshold

        df['stochrsi_sell'] = sell
        df['stochrsi_buy'] = buy

        # update data for execution of order
        global stochrsi_4h_sell
        global stochrsi_4h_buy
        stochrsi_4h_sell[symbol] = sell
        stochrsi_4h_buy[symbol] = buy

        print(f'\n----------- {symbol} Stochrsi Signal Analysis (4 hours) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_supertrend(exchange, symbol: str)->None:
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

        global supertrend_up
        supertrend_up[symbol] = df.iloc[-1]['in_uptrend']

        global supertrend_buy
        supertrend_buy[symbol] = df.iloc[-1]['in_uptrend']

        if supertrend_buy[symbol] == True:
            supertrend_sell_quota[symbol] = 4000000

        global supertrend_sell
        supertrend_sell[symbol] = (df.iloc[-2]['in_uptrend'] == True) and (df.iloc[-1]['in_uptrend'] == False)

        if supertrend_sell[symbol] == True:
            supertrend_sell_iter[symbol] = 1

    except Exception as e:
        print("Exception : ", str(e))

## Todo monitor bitcoin trading for market sentiment estimation 
##    +1% change in 30 minutes : buy or binance call x5 order  
##    -1% change in 30 minutes : sell or binance put x5 order
## utilize some action after this event 
'''
def analyze_bitcoin_30m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='30m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0) 

        # Get the latest value
        current_stochrsi_k = df['stochrsi_k'].iloc[-1]
        current_stochrsi_d = df['stochrsi_d'].iloc[-1]

        # Stoch rsi cross-over strategy
        sell = current_stochrsi_k < current_stochrsi_d and current_stochrsi_k > overbought_threshold
        buy = current_stochrsi_k > current_stochrsi_d and current_stochrsi_k < oversold_threshold 

        # update data for execution of order
        global stochrsi_30m_sell
        global stochrsi_30m_buy
        stochrsi_30m_sell[symbol] = sell
        stochrsi_30m_buy[symbol] = buy

        df['stochrsi_sell'] = sell
        df['stochrsi_buy'] = buy

        print(f'\n----------- {symbol} Stochrsi Signal Analysis (30 Minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))
'''

def show_orderbook(orderbook):
        print("\n------------Getting order book -----------")
        pprint(orderbook)

def bollinger_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price  = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount = round((bb_trading_amount)/ price, 5)
        resp   = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Bollinger sell : {symbol}, price={price}, amount={amount},\
                     threshold={bollinger_threshold[symbol]}, width={bollinger_width[symbol]}, mfi_4h={mfi_4h[symbol]}")

    except Exception as e:
        print("Exception : ", str(e))

def bollinger_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (bb_trading_amount ):
            amount = (bb_trading_amount)
        else:
            logging.info(f"Cancel bollinger buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_sell_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"Bollinger buy :{symbol}, price={price}, amount = {amount}, \
                     threshold={bollinger_threshold[symbol]}, width={bollinger_width[symbol]}, mfi_4h={mfi_4h[symbol]}")

    except Exception as e:
        print("Exception : ", str(e))

def rsi_3m_scalping_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((rsi_3m_scalping_sell_amount)/price, 3)
        resp      =exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"MFI(5m) scalping sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def rsi_3m_scalping_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (rsi_3m_scalping_buy_amount ):
            amount = (rsi_3m_scalping_buy_amount)
        else:
            logging.info(f"Cancel RSI(3m) buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"RSI(3m) scalping buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def mfi_5m_scalping_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((mfi_5m_scalping_sell_amount)/price, 3)
        resp      =exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"MFI(5m) scalping sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def mfi_5m_scalping_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (mfi_5m_scalping_buy_amount ):
            amount = (mfi_5m_scalping_buy_amount)
        else:
            logging.info(f"Cancel MFI(5m) buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"MFI(5m) scalping buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def mfi_4h_scalping_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((mfi_4h_scalping_sell_amount)/price, 3)
        resp      =exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"MFI(4h) scalping sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def mfi_4h_scalping_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (mfi_4h_scalping_buy_amount ):
            amount = (mfi_4h_scalping_buy_amount)
        else:
            logging.info(f"Cancel MFI buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"MFI(4h) scalping buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_5m_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((stochrsi_5m_sell_amount)/price, 3)
        resp      = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Stochrsi(5m) Sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_5m_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)

        amount = 0.0
        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW > stochrsi_5m_buy_amount:
            amount = stochrsi_5m_buy_amount
        else:
            logging.info(f"Cancel STOCHRSI (5m) buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"STOCHRSI(5m) buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_4h_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((stochrsi_4h_sell_amount)/price, 3)
        resp      = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Stochrsi(4h) Sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))


def stochrsi_4h_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        amount = 0.0
        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW > (stochrsi_4h_buy_amount ):
            amount = (stochrsi_4h_buy_amount) 
        else:
            logging.info(f"Cancel strochrsi 30 minutes buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"STOCHRSI(4h) Buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))


def supertrend_sell_update(symbol: str):
    global supertrend_sell_amount
    supertrend_sell_amount[symbol] = supertrend_sell_quota[symbol] / pow(1.5, supertrend_sell_iter[symbol])

def supertrend_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((supertrend_sell_amount)/price, 3)
        resp      = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Supertrend Sell order placed for {symbol} at price: {price}, amount = {amount}")

        global supertrend_sell_iter 
        supertrend_sell_iter[symbol] = supertrend_sell_iter[symbol] + 1

        supertrend_sell_amount_update(symbol)

        if supertrend_sell_iter > 10 :
            global supertrend_sell
            supertrend_sell[symbol] = False

    except Exception as e:
        print("Exception : ", str(e))

def supertrend_buy_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)

        amount = 0.0

        free_KRW = exchange.fetchBalance()['KRW']['free']
        if free_KRW > (supertrend_buy_amount ):
            amount = (supertrend_buy_amount)
        else:
            logging.info(f"Cancel supertrend buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        logging.info(f"Supertrend Buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def execute_bollinger_order(exchange, symbol: str)->None:

    sell = bollinger_sell[symbol]
    buy  = bollinger_buy[symbol]

    if buy:
       bollinger_buy_coin(exchange, symbol)
    elif sell:
       bollinger_sell_coin(exchange, symbol)

    global iterations
    iterations[symbol] = iterations[symbol] + 1

    if (iterations[symbol] % 15 == 0):
       reset_bollinger_order(symbol)

def execute_rsi_3m_buy_order(exchange, symbol: str)->None:
    buy = rsi_3m_scalping_buy[symbol]

    if buy:
        rsi_3m_scalping_buy_coin(exchange, symbol)

def execute_rsi_3m_sell_order(exchange, symbol: str)->None:
    sell = rsi_3m_scalping_sell[symbol]

    if sell:
       rsi_3m_scalping_sell_coin(exchange, symbol)

def execute_mfi_5m_buy_order(exchange, symbol: str)->None:
    buy = mfi_5m_scalping_buy[symbol]

    if buy:
        mfi_5m_scalping_buy_coin(exchange, symbol)

def execute_mfi_5m_sell_order(exchange, symbol: str)->None:
    sell = mfi_5m_scalping_sell[symbol]

    if sell:
       mfi_5m_scalping_sell_coin(exchange, symbol)

def execute_mfi_4h_scapling_sell(exchange, symbol: str)->None:
    buy = mfi_4h_scalping_sell[symbol]

    if buy:
        mfi_4h_scalping_sell_coin(exchange, symbol)

def execute_mfi_4h_scalping_buy(exchange, symbol: str)->None:
    sell = mfi_4h_scalping_buy[symbol]

    if sell:
       mfi_4h_scalping_buy_coin(exchange, symbol)

def execute_stochrsi_5m_sell(exchange, symbol: str)->None:
    sell = stochrsi_5m_sell[symbol]

    if sell:
       stochrsi_5m_sell_coin(exchange, symbol)

def execute_stochrsi_5m_buy(exchange, symbol: str)->None:
    buy = stochrsi_5m_buy[symbol]

    if buy:
        stochrsi_5m_buy_coin(exchange, symbol)

def execute_stochrsi_4h_sell(exchange, symbol: str)->None:
    sell = stochrsi_4h_sell[symbol] 

    if sell:
       stochrsi_4h_sell_coin(exchange, symbol)

def execute_stochrsi_4h_buy(exchange, symbol: str)->None:
    buy = stochrsi_4h_buy[symbol]

    if buy:
        stochrsi_4h_buy_coin(exchange, symbol)

def execute_supertrend_sell(exchange, symbol: str):
    sell = supertrend_sell[symbol]

    if sell:
        supertrend_sell_coin(exchange, symbol)

def execute_supertrend_buy(exchange, symbol:str):
    buy = supertrend_buy[symbol]

    if buy:
        supertrend_buy_coin(exchange, symbol)

def monitor(symbols : list[str]):
    print("\n---------------- buy/sell order summary -----------------")

    column_name= ["Symbol","Supertrend Up", "Buy", "Sell", "MFI(5m) Buy", "MFI(5m) Sell", "MFI(4h) Buy", "MFI(4h) Sell"]
    orders = pd.DataFrame(columns = column_name)

    for s in symbols:
        orders.loc[len(orders)] = [s, supertrend_up[s], bollinger_buy[s], bollinger_sell[s],\
                                   mfi_5m_scalping_buy[s], mfi_5m_scalping_sell[s], \
                                   mfi_4h_scalping_buy[s], mfi_4h_scalping_sell[s]]
    pprint(orders)

def monitor_balance(exchange):
    try:
        print("\n---------------- fetch balance result  -----------------")
        balance = exchange.fetchBalance()
        pprint(balance)

    except Exception as e:
        print("Exception : ", str(e))

def analyze_covariance(exchange, currencies)->None:
    try:
        print("------------- analyze covariance------------")

        daily_price = pd.DataFrame()
        daily_return = pd.DataFrame()
        annual_return = pd.DataFrame()

        for symbol in currencies:

            ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe='1d')
            df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            daily_price[symbol]= df['close']
            daily_return[symbol] = daily_price[symbol].pct_change()

        annual_return = daily_return.mean() * 365

        print("\n---------------daily return -------------")
        pprint(daily_return)
        print("\n---------------daily covariance -------------")
        daily_cov = daily_return.cov()
        pprint(daily_cov)
        print("\n---------------annual return -------------")
        pprint(annual_return)

        print("\n-------------- annual convariance -----------")
        annual_cov = daily_cov*365

        pprint(annual_cov) 

        portfolio_return =[]
        portfolio_risk = []
        portfolio_weights = []

        for _ in range(20000):
            weights = np.random.random(len(currencies))
            weights /= np.sum(weights)

            returns = np.dot(weights, annual_return)
            risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))

            portfolio_return.append(returns)
            portfolio_risk.append(risk)
            portfolio_weights.append(weights)

        portfolio = {'Returns': portfolio_return, 'Risk': portfolio_risk}
        for i, s in enumerate(currencies):
            portfolio[s] = [weight[i] for weight in portfolio_weights]

        df = pd.DataFrame(portfolio)
        df - df[['Returns', 'Risk'] + [s for s in currencies]]

        pprint(df)

        df.plot.scatter(x = 'Risk', y ='Returns', figsize=(9, 8), grid=True)
        plt.title("Efficient Frontier")
        plt.show()

    except Exception as e:
        print("Exception: ", str(e))

def init_supertrend_quota(symbols):

    global supertrend_sell_quota 

    for x in symbols:
        supertrend_sell_quota[x]= 4000000

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

def init_binance():
    print('\n---------------Binance Exchange Initilization-------------------------')
    exchange = ccxt.binance(config={
            'apiKey':binance_key['apiKey'],
            'secret':binance_key['secret'],
            'enableRateLimit': True,
    })
    return exchange


if __name__=='__main__':

    # Configure logging
    logging.basicConfig(filename="./trading.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    exchange = init_upbit()
    binance = init_binance()

    #define symbols 
    doge = "DOGE/KRW"
    xrp  = "XRP/KRW"
    sol  = "SOL/KRW"
    btc  = "BTC/KRW"
    eth  = "ETH/KRW"

    symbols= [doge, xrp, sol, btc, eth]
    init_supertrend_quota(symbols)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, doge)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, doge)
    schedule.every(30).seconds.do(analyze_stochrsi_5m, exchange, doge)
    schedule.every(30).seconds.do(analyze_stochrsi_4h, exchange, doge)
    schedule.every(30).seconds.do(analyze_mfi_signals_5m, exchange, doge)
    schedule.every(30).seconds.do(analyze_rsi_signals_3m, exchange, doge)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, doge)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, xrp)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_stochrsi_5m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_stochrsi_4h, exchange, xrp)
    schedule.every(30).seconds.do(analyze_mfi_signals_5m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_rsi_signals_3m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, xrp)
 
    schedule.every(30).seconds.do(analyze_signals_1d, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, sol)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, sol)
    schedule.every(30).seconds.do(analyze_stochrsi_5m, exchange, sol)
    schedule.every(30).seconds.do(analyze_stochrsi_4h, exchange, sol)
    schedule.every(30).seconds.do(analyze_mfi_signals_5m, exchange, sol)
    schedule.every(30).seconds.do(analyze_rsi_signals_3m, exchange, sol)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, sol)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, btc)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, btc)
    schedule.every(30).seconds.do(analyze_stochrsi_5m, exchange, btc)
    schedule.every(30).seconds.do(analyze_stochrsi_4h, exchange, btc)
    schedule.every(30).seconds.do(analyze_mfi_signals_5m, exchange, btc)
    schedule.every(30).seconds.do(analyze_rsi_signals_3m, exchange, btc)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, btc)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, eth)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, eth)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, eth)
    schedule.every(30).seconds.do(analyze_stochrsi_5m, exchange, eth)
    schedule.every(30).seconds.do(analyze_stochrsi_4h, exchange, eth)
    schedule.every(30).seconds.do(analyze_mfi_signals_5m, exchange, eth)
    schedule.every(30).seconds.do(analyze_rsi_signals_3m, exchange, eth)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, eth)

    #bollinger band order every 5 minutes with bollinger(5m) analysis
    schedule.every(5).minutes.do(execute_bollinger_order, exchange, doge)
    schedule.every(5).minutes.do(execute_bollinger_order, exchange, xrp)
    schedule.every(5).minutes.do(execute_bollinger_order, exchange, sol)
    schedule.every(5).minutes.do(execute_bollinger_order, exchange, btc)
    schedule.every(5).minutes.do(execute_bollinger_order, exchange, eth)

    # every 3 minute scalping with RSI(3m)
    schedule.every(3).minutes.do(execute_rsi_3m_buy_order, exchange, doge)
    schedule.every(3).minutes.do(execute_rsi_3m_buy_order, exchange, xrp)
    schedule.every(3).minutes.do(execute_rsi_3m_buy_order, exchange, sol)
    schedule.every(3).minutes.do(execute_rsi_3m_buy_order, exchange, eth)
    schedule.every(3).minutes.do(execute_rsi_3m_buy_order, exchange, btc)
    schedule.every(3).minutes.do(execute_rsi_3m_sell_order, exchange, doge)
    schedule.every(3).minutes.do(execute_rsi_3m_sell_order, exchange, xrp)
    schedule.every(3).minutes.do(execute_rsi_3m_sell_order, exchange, sol)
    schedule.every(3).minutes.do(execute_rsi_3m_sell_order, exchange, eth)
    schedule.every(3).minutes.do(execute_rsi_3m_sell_order, exchange, btc)

    # every 5 minute scalping with MFI(5m)
    schedule.every(5).minutes.do(execute_mfi_5m_buy_order, exchange, doge)
    schedule.every(5).minutes.do(execute_mfi_5m_buy_order, exchange, xrp)
    schedule.every(5).minutes.do(execute_mfi_5m_buy_order, exchange, sol)
    schedule.every(5).minutes.do(execute_mfi_5m_buy_order, exchange, btc)
    schedule.every(5).minutes.do(execute_mfi_5m_buy_order, exchange, eth)
    schedule.every(5).minutes.do(execute_mfi_5m_sell_order, exchange, doge)
    schedule.every(5).minutes.do(execute_mfi_5m_sell_order, exchange, xrp)
    schedule.every(5).minutes.do(execute_mfi_5m_sell_order, exchange, sol)
    schedule.every(5).minutes.do(execute_mfi_5m_sell_order, exchange, btc)
    schedule.every(5).minutes.do(execute_mfi_5m_sell_order, exchange, eth)

    # mfi 4 hour scalping
    schedule.every(1).hours.do(execute_mfi_4h_scapling_sell, exchange, doge)
    schedule.every(1).hours.do(execute_mfi_4h_scalping_buy, exchange, doge)
    schedule.every(1).hours.do(execute_mfi_4h_scalping_buy, exchange, xrp)
    schedule.every(1).hours.do(execute_mfi_4h_scapling_sell, exchange, xrp)
    schedule.every(1).hours.do(execute_mfi_4h_scalping_buy, exchange, sol)
    schedule.every(1).hours.do(execute_mfi_4h_scapling_sell, exchange, sol)
    schedule.every(1).hours.do(execute_mfi_4h_scalping_buy, exchange, btc)
    schedule.every(1).hours.do(execute_mfi_4h_scapling_sell, exchange, btc)
    schedule.every(1).hours.do(execute_mfi_4h_scalping_buy, exchange, eth)
    schedule.every(1).hours.do(execute_mfi_4h_scapling_sell, exchange, eth)

    #stochrsi (5m) based order every 5 minutes
    schedule.every(5).minutes.do(execute_stochrsi_5m_buy, exchange, doge)
    schedule.every(5).minutes.do(execute_stochrsi_5m_buy, exchange, xrp)
    schedule.every(5).minutes.do(execute_stochrsi_5m_buy, exchange, sol)
    schedule.every(5).minutes.do(execute_stochrsi_5m_buy, exchange, btc)
    schedule.every(5).minutes.do(execute_stochrsi_5m_buy, exchange, eth)
    schedule.every(5).minutes.do(execute_stochrsi_5m_sell, exchange, doge)
    schedule.every(5).minutes.do(execute_stochrsi_5m_sell, exchange, xrp)
    schedule.every(5).minutes.do(execute_stochrsi_5m_sell, exchange, sol)
    schedule.every(5).minutes.do(execute_stochrsi_5m_sell, exchange, btc)
    schedule.every(5).minutes.do(execute_stochrsi_5m_sell, exchange, eth)

    #stochrsi (4h) based order every 1 hour 
    schedule.every(1).hours.do(execute_stochrsi_4h_buy, exchange, doge)
    schedule.every(1).hours.do(execute_stochrsi_4h_buy, exchange, xrp)
    schedule.every(1).hours.do(execute_stochrsi_4h_buy, exchange, sol)
    schedule.every(1).hours.do(execute_stochrsi_4h_buy, exchange, btc)
    schedule.every(1).hours.do(execute_stochrsi_4h_buy, exchange, eth)
    schedule.every(1).hours.do(execute_stochrsi_4h_sell, exchange, doge)
    schedule.every(1).hours.do(execute_stochrsi_4h_sell, exchange, xrp)
    schedule.every(1).hours.do(execute_stochrsi_4h_sell, exchange, sol)
    schedule.every(1).hours.do(execute_stochrsi_4h_sell, exchange, btc)
    schedule.every(1).hours.do(execute_stochrsi_4h_sell, exchange, eth)

    #supertrend order every 3 hours
    schedule.every(3).hours.do(execute_supertrend_buy, exchange, doge)
    schedule.every(3).hours.do(execute_supertrend_buy, exchange, xrp)
    schedule.every(3).hours.do(execute_supertrend_buy, exchange, sol)
    schedule.every(3).hours.do(execute_supertrend_buy, exchange, btc)
    schedule.every(3).hours.do(execute_supertrend_buy, exchange, eth)
    schedule.every(3).hours.do(execute_supertrend_sell, exchange, doge)
    schedule.every(3).hours.do(execute_supertrend_sell, exchange, xrp)
    schedule.every(3).hours.do(execute_supertrend_sell, exchange, sol)
    schedule.every(3).hours.do(execute_supertrend_sell, exchange, btc)
    schedule.every(3).hours.do(execute_supertrend_sell, exchange, eth)

    # monitoring every 30 seconds
    schedule.every(30).seconds.do(monitor, symbols)
    schedule.every(30).seconds.do(monitor_balance, exchange)

    while True:
        schedule.run_pending()
        time.sleep(0.01)
