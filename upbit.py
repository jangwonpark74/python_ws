import ccxt
import talib
import time
import schedule
import pandas as pd
import logging 
import time

from pprint import pprint
from collections import defaultdict

from conf import key
from conf import binance_key

# Bollinger band buy/sell order 
buy_order = defaultdict(bool)
sell_order = defaultdict(bool)

# MFI scalping 3 minute
scalping_buy = defaultdict(bool)
scalping_sell= defaultdict(bool)

# Bollinger band analysis based buy, sell amount
bb_trading_amount = 2000000

# MFI 5 minute scalping amount 
mfi_scalping_sell_amount = 3000000
mfi_scalping_buy_amount  = 3000000

# 30m stochrsi amount 
stochrsi_30m_sell_amount = 2000000
stochrsi_30m_buy_amount  = 2000000

# MFI 3 minutes for scalping
mfi_3m = defaultdict(float)

# MFI 4 hour for volatility analysis
mfi_4h = defaultdict(float)
mfi_1d = defaultdict(float)

# MFI high low threshold
mfi_high_threshold = 83
mfi_low_threshold = 25
rsi_low_threshold = 25

# Define parameters for Stochastic RSI
overbought_threshold = 80 
oversold_threshold = 23

# Stoch RSI sell buy every 15 minutes
stochrsi_15m_sell = defaultdict(bool)
stochrsi_15m_buy = defaultdict(bool)

# Stoch RSI sell buy every 30 minutes 
stochrsi_30m_sell = defaultdict(bool)
stochrsi_30m_buy = defaultdict(bool)

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

#pd.set_option('display.max_rows', None)

def reset_sell_buy_order(symbol: str):
    global sell_order
    global buy_order
    sell_order[symbol] = False
    buy_order[symbol] = False

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
        df_1d = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1d['datetime'] = pd.to_datetime(df_1d['datetime'], utc=True, unit='ms')
        df_1d['datetime'] = df_1d['datetime'].dt.tz_convert("Asia/Seoul")
        df_1d['bollinger_upper'], df_1d['bollinger_middle'], df_1d['bollinger_lower'] = talib.BBANDS(df_1d['close'])
        df_1d['mfi']             = round( talib.MFI(df_1d['high'], df_1d['low'], df_1d['close'], df_1d['volume'], timeperiod=14), 1)

        df_1d['bollinger_width'] = round(((df_1d['bollinger_upper'] - df_1d['bollinger_lower'])/df_1d['bollinger_middle']) * 100, 1)
        df_1d['bollinger_upper'] = round(df_1d['bollinger_upper'], 1)
        df_1d['bollinger_middle']= round(df_1d['bollinger_middle'], 1)
        df_1d['bollinger_lower'] = round(df_1d['bollinger_lower'], 1)
    
        print(f'\n----------------------- {symbol} Signal Analysis ( 1 day ) -----------------------------')
        pprint(df_1d.iloc[-1])

        # daily mfi update 
        global mfi_1d
        mfi = round (df_1d['mfi'].iloc[-1], 2)
        mfi_1d[symbol] = mfi
 
        print(f'\nSymbol: {symbol} MFI(1d, 14) = {mfi_1d[symbol]}')

    except Exception as e:
        print("Exception : ", str(e))

def analyze_signals_4h(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = round(talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 1)

        global mfi_4h
        mfi_4h[symbol] = df['mfi'].iloc[-1]

        print(f'\n----------- {symbol} MFI analysis (4 hour) --------------')
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
        mfi = mfi_4h[symbol] 
        bb_width = df['bollinger_width'].iloc[-1]
        
        # sell, buy condition check
        global sell_order
        sell = (df['high'].iloc[-1] > df['bollinger_upper'].iloc[-1]) and (bb_width > calc_volatility(mfi)) 
        sell_order[symbol] = sell
        df['bollinger_sell'] = sell

        global buy_order
        buy = (df['low'].iloc[-1] < df['bollinger_lower'].iloc[-1]) and  (bb_width > calc_volatility(mfi))
        buy_order[symbol] = buy 
        df['bollinger_buy'] = buy

        print(f'\n----------- {symbol} Bollinger Sell/Buy and Volatiltiy Analysis (15 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_stochrsi_15m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        # 
        df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=3, fastd_period=3, fastd_matype=0) 

        # Get the latest value
        current_stochrsi_k = df['stochrsi_k'].iloc[-1]
        current_stochrsi_d = df['stochrsi_d'].iloc[-1]

        # Stoch rsi cross-over strategy
        sell = current_stochrsi_k < current_stochrsi_d and current_stochrsi_k > overbought_threshold
        buy = current_stochrsi_k > current_stochrsi_d and current_stochrsi_k < oversold_threshold 

        # update data for execution of order
        global stochrsi_15m_sell
        global stochrsi_15m_buy
        stochrsi_15m_sell[symbol] = sell
        stochrsi_15m_buy[symbol] = buy

        df['stochrsi_sell'] = sell
        df['stochrsi_buy'] = buy

        print(f'\n----------- {symbol} STOCHRSI Signal Analysis (15 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def analyze_stochrsi_30m(exchange, symbol: str)->None:
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

def analyze_mfi_signals_3m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='3m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = round(talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 2)
        df['rsi']      = round(talib.RSI(df['close'], timeperiod=14), 2)

        # Scalping based on 3 minutes MFI  
        mfi = df['mfi'].iloc[-1]
        rsi = df['rsi'].iloc[-1]

        sell = mfi > mfi_high_threshold
        buy  = (mfi < mfi_low_threshold) | (rsi < rsi_low_threshold)

        global mfi_3m
        mfi_3m[symbol] = mfi

        # update data for execution of order
        global scalping_sell
        global scalping_buy
        scalping_sell[symbol] = sell
        scalping_buy[symbol] = buy

        # store information for dispaly
        df['mfi_3m_scalping_sell'] = sell
        df['mfi_rsi_3m_scalping_buy']  = buy

        print(f'\n----------- {symbol} Signal Analysis (3 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def show_orderbook(orderbook):
        print("\n------------Getting order book -----------")
        pprint(orderbook)

def sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((bb_trading_amount)/ price, 5)
        resp = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Bollinger sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def mfi_scalping_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((mfi_scalping_sell_amount)/price, 3)
        resp      =exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"MFI scalping sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_15m_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((mfi_scalping_sell_amount)/price, 3)
        resp      = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Stochrsi 15 minutes Sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_30m_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        price     = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((stochrsi_30m_sell_amount)/price, 3)
        resp      = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        logging.info(f"Stochrsi 30 Minutes Sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))


def buy_coin(exchange,symbol: str)->None:
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
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"Bollinger buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def mfi_scalping_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (mfi_scalping_buy_amount ):
            amount = (mfi_scalping_buy_amount) 
        else:
            logging.info(f"Cancel MFI buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)
        
        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"MFI scalping buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))


def stochrsi_15m_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)

        amount = 0.0
        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW > (mfi_scalping_buy_amount ):
            amount = (mfi_scalping_buy_amount) 
        else:
            logging.info(f"Cancel STOCHRSI 15 minutes buy for low balance {symbol} free KRW = {free_KRW}")
            return

        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)

        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"STOCHRSI 15 minutes buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def stochrsi_30m_buy_coin(exchange,symbol: str)->None:
    try:
        orderbook = exchange.fetch_order_book(symbol)
        amount = 0.0
        free_KRW = exchange.fetchBalance()['KRW']['free']

        if free_KRW > (stochrsi_30m_buy_amount ):
            amount = (stochrsi_30m_buy_amount) 
        else:
            logging.info(f"Cancel strochrsi 30 minutes buy for low balance {symbol} free KRW = {free_KRW}")
            return
        
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)
        
        show_orderbook(orderbook)
        price = round(orderbook['asks'][0][0], 1)
        logging.info(f"STOCHRSI 30 minutes buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))


def supertrend_sell_update(symbol: str):
    global supertrend_sell_amount
    supertrend_sell_amount[symbol] = supertrend_sell_quota[symbol] / pow(1.5, supertrend_sell_iter[symbol])
 
def supertrend_sell_coin(exchange, symbol: str):
    try:
        orderbook = exchange.fetch_order_book(symbol)
        amount    = round((supertrend_sell_amount)/price, 3)
        resp      = exchange.create_market_sell_order(symbol=symbol, amount = amount )

        show_orderbook(orderbook)
        price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        logging.info(f"Supertrend sell order placed for {symbol} at price: {price}, amount = {amount}")

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
        logging.info(f"Supertrend buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def execute_order(exchange, symbol: str)->None:
    
    sell = sell_order[symbol]
    buy  = buy_order[symbol]

    if buy:
       buy_coin(exchange, symbol)
    elif sell:
       sell_coin(exchange, symbol)

    global iterations
    iterations[symbol] = iterations[symbol] + 1

    if (iterations[symbol] % 15 == 0):
       reset_sell_buy_order(symbol)

def execute_mfi_scalping_buy(exchange, symbol: str)->None:
    sell = scalping_sell[symbol] 

    if sell:
       mfi_scalping_sell_coin(exchange, symbol)

def execute_mfi_scapling_sell(exchange, symbol: str)->None:
    buy = scalping_buy[symbol]

    if buy:
        mfi_scalping_buy_coin(exchange, symbol)

def execute_stochrsi_15m_buy(exchange, symbol: str)->None:
    sell = stochrsi_15m_sell[symbol] 

    if sell:
       stochrsi_15m_sell_coin(exchange, symbol)

def execute_stochrsi_15m_sell(exchange, symbol: str)->None:
    buy = stochrsi_15m_buy[symbol]

    if buy:
        stochrsi_15m_buy_coin(exchange, symbol)

def execute_stochrsi_30m_buy(exchange, symbol: str)->None:
    sell = stochrsi_30m_sell[symbol] 

    if sell:
       stochrsi_30m_sell_coin(exchange, symbol)

def execute_stochrsi_30m_sell(exchange, symbol: str)->None:
    buy = stochrsi_30m_buy[symbol]

    if buy:
        stochrsi_30m_buy_coin(exchange, symbol)

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

    column_name= ["Symbol","Supertrend Up", "Buy", "Sell", "Scalping Buy", "Scalping Sell"]
    orders = pd.DataFrame(columns = column_name)

    for s in symbols:
        orders.loc[len(orders)] = [s, supertrend_up[s], buy_order[s], sell_order[s], scalping_buy[s],scalping_sell[s]]
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
    schedule.every(30).seconds.do(analyze_stochrsi_15m, exchange, doge)
    schedule.every(30).seconds.do(analyze_stochrsi_30m, exchange, doge)
    schedule.every(30).seconds.do(analyze_mfi_signals_3m, exchange, doge)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, doge)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, xrp)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_stochrsi_15m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_mfi_signals_3m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, xrp)
    
    schedule.every(30).seconds.do(analyze_signals_1d, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, sol)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, sol)
    schedule.every(30).seconds.do(analyze_stochrsi_15m, exchange, sol)
    schedule.every(30).seconds.do(analyze_stochrsi_30m, exchange, sol)
    schedule.every(30).seconds.do(analyze_mfi_signals_3m, exchange, sol)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, sol)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, btc)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, btc)
    schedule.every(30).seconds.do(analyze_stochrsi_15m, exchange, btc)
    schedule.every(30).seconds.do(analyze_stochrsi_30m, exchange, btc)
    schedule.every(30).seconds.do(analyze_mfi_signals_3m, exchange, btc)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, btc)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, eth)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, eth)
    schedule.every(30).seconds.do(analyze_bb_signals_15m, exchange, eth)
    schedule.every(30).seconds.do(analyze_stochrsi_15m, exchange, eth)
    schedule.every(30).seconds.do(analyze_stochrsi_30m, exchange, eth)
    schedule.every(30).seconds.do(analyze_mfi_signals_3m, exchange, eth)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, eth)

    #bollinger band order every 1minutes with 15minutes analysis
    schedule.every(1).minutes.do(execute_order, exchange, doge)
    schedule.every(1).minutes.do(execute_order, exchange, xrp)
    schedule.every(1).minutes.do(execute_order, exchange, sol)
    schedule.every(1).minutes.do(execute_order, exchange, btc)
    schedule.every(1).minutes.do(execute_order, exchange, eth)

    # mfi 3 minute scalping
    schedule.every(3).minutes.do(execute_mfi_scapling_sell, exchange, doge)
    schedule.every(3).minutes.do(execute_mfi_scalping_buy, exchange, doge)
    schedule.every(3).minutes.do(execute_mfi_scalping_buy, exchange, xrp)
    schedule.every(3).minutes.do(execute_mfi_scapling_sell, exchange, xrp)
    schedule.every(3).minutes.do(execute_mfi_scalping_buy, exchange, sol)
    schedule.every(3).minutes.do(execute_mfi_scapling_sell, exchange, sol)
    schedule.every(3).minutes.do(execute_mfi_scalping_buy, exchange, btc)
    schedule.every(3).minutes.do(execute_mfi_scapling_sell, exchange, btc)
    schedule.every(3).minutes.do(execute_mfi_scalping_buy, exchange, eth)
    schedule.every(3).minutes.do(execute_mfi_scapling_sell, exchange, eth)

    #stochrsi based order every 5 minutes
    schedule.every(5).minutes.do(execute_stochrsi_15m_buy, exchange, doge)
    schedule.every(5).minutes.do(execute_stochrsi_15m_buy, exchange, xrp)
    schedule.every(5).minutes.do(execute_stochrsi_15m_buy, exchange, sol)
    schedule.every(5).minutes.do(execute_stochrsi_15m_buy, exchange, btc)
    schedule.every(5).minutes.do(execute_stochrsi_15m_buy, exchange, eth)
    schedule.every(5).minutes.do(execute_stochrsi_15m_sell, exchange, doge)
    schedule.every(5).minutes.do(execute_stochrsi_15m_sell, exchange, xrp)
    schedule.every(5).minutes.do(execute_stochrsi_15m_sell, exchange, sol)
    schedule.every(5).minutes.do(execute_stochrsi_15m_sell, exchange, btc)
    schedule.every(5).minutes.do(execute_stochrsi_15m_sell, exchange, eth)

    #stochrsi based order every 30 minutes 
    schedule.every(30).minutes.do(execute_stochrsi_30m_buy, exchange, doge)
    schedule.every(30).minutes.do(execute_stochrsi_30m_buy, exchange, xrp)
    schedule.every(30).minutes.do(execute_stochrsi_30m_buy, exchange, sol)
    schedule.every(30).minutes.do(execute_stochrsi_30m_buy, exchange, btc)
    schedule.every(30).minutes.do(execute_stochrsi_30m_buy, exchange, eth)
    schedule.every(30).minutes.do(execute_stochrsi_30m_sell, exchange, doge)
    schedule.every(30).minutes.do(execute_stochrsi_30m_sell, exchange, xrp)
    schedule.every(30).minutes.do(execute_stochrsi_30m_sell, exchange, sol)
    schedule.every(30).minutes.do(execute_stochrsi_30m_sell, exchange, btc)
    schedule.every(30).minutes.do(execute_stochrsi_30m_sell, exchange, eth)

    #supertrend order every 2 hours
    schedule.every(2).hours.do(execute_supertrend_sell, exchange, doge)
    schedule.every(2).hours.do(execute_supertrend_buy, exchange, doge)
    schedule.every(2).hours.do(execute_supertrend_sell, exchange, xrp)
    schedule.every(2).hours.do(execute_supertrend_buy, exchange, xrp)
    schedule.every(2).hours.do(execute_supertrend_sell, exchange, sol)
    schedule.every(2).hours.do(execute_supertrend_buy, exchange, sol)
    schedule.every(2).hours.do(execute_supertrend_sell, exchange, btc)
    schedule.every(2).hours.do(execute_supertrend_buy, exchange, btc)
    schedule.every(2).hours.do(execute_supertrend_sell, exchange, eth)
    schedule.every(2).hours.do(execute_supertrend_buy, exchange, eth)

    # monitoring every 30 seconds
    schedule.every(30).seconds.do(monitor, symbols)
    schedule.every(30).seconds.do(monitor_balance, exchange)

    while True:
        schedule.run_pending()
        time.sleep(0.01)
