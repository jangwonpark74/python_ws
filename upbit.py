import ccxt
import talib
import time
import schedule
import pandas as pd
import logging 

from conf import key
from pprint import pprint
from collections import defaultdict

# Bollinger band buy/sell order 
buy_order = defaultdict(bool)
sell_order = defaultdict(bool)

# MFI scalping 3 minute
scalping_buy = defaultdict(bool)
scalping_sell= defaultdict(bool)

# Bollinger band analysis based buy, sell amount
bb_trading_amount = 1000000

# MFI and RSI analysis based scalping amount 
scalping_sell_amount = 3000000
scalping_buy_amount  = 3000000

# MFI 4hour for volatility analysis
mfi_4h = defaultdict(float)
mfi_1d = defaultdict(float)

# Global variable to keep the count for max 15 minutes continue for order
iterations = defaultdict(int)

# Global variable to keep supertrend sell count
supertrend_sell_iter = defaultdict(int)

# Current supertrend 
supertrend_up = defaultdict(bool)

# Supertrend buy, sell
supertrend_buy = defaultdict(bool)
supertrend_sell = defaultdict(bool)

# supertrend buy amount at every 4 hour
supertrend_buy_amount = 1000000

# supertrend sell one time 
supertrend_sell_quota = defaultdict(float) 
supertrend_sell_amount = defaultdict(float)

pd.set_option('display.max_rows', None)


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

def reset_sell_buy_order(symbol: str):
    global sell_order
    global buy_order
    sell_order[symbol] = False
    buy_order[symbol] = False

def reset_sell_buy_order_4h(symbol : str):
    global sell_order_4h
    global buy_order_4h
    sell_order_4h[symbol] = False
    buy_order_4h[symbol] = False

def calc_volatility(x: float) -> float:
    volatility = round(-0.0012 * x * x + 0.12 * x +0.5, 2)
    return volatility

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
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h', limit= 120)
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        df['high-low'] = df['high'] - df['low']
        df['pc']       = df['close'].shift(1)
        df['high-pc']  = abs(df['high'] - df['pc'])
        df['low-pc']   = abs(df['low'] -df['pc'])
        
        df['tr'] = df[['high-low', 'high-pc', 'low-pc']].max(axis=1) 

        period = 14
        multiplier = 2.0

        df['atr'] = df['tr'].rolling(period).mean() 

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
            supertrend_sell_quota[symbol] = 400000

        global supertrend_sell
        supertrend_sell[symbol] = (df.iloc[-2]['in_uptrend'] == True) and (df.iloc[-1]['in_uptrend'] == False)

        if supertrend_sell[symbol] == True:
            supertrend_sell_iter[symbol] = 1 

    except Exception as e:
        print("Exception : ", str(e))


def analyze_signals_15m(exchange, symbol: str)->None:
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


def analyze_signals_3m(exchange, symbol: str)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='3m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['mfi']      = round(talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 2)

        # Scalping based on 3 minute MFI and RSI 
        mfi_3m = df['mfi'].iloc[-1]

        sell = ( mfi_3m > 80 ) 
        buy  = (mfi_3m < 25) 
        df['scalping_sell'] = sell 
        df['scalping_buy'] = buy 
        
        global scalping_sell 
        global scalping_buy

        scalping_sell[symbol] = sell 
        scalping_buy[symbol] = buy 

        print(f'\n----------- {symbol} Signal Analysis (3 minutes) --------------')
        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))

def sell_coin(exchange, symbol: str):
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol)
        pprint(orderbook)

        price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((bb_trading_amount)/ price, 5)

        print("\n------------ Make a sell order-----------")
        print(f'{symbol} price : {price}, sell amount = {amount}')  
        resp = exchange.create_market_sell_order(symbol=symbol, amount = amount )
        pprint(resp)

        logging.info(f"Sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def scalping_sell_coin(exchange, symbol: str):
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol)
        pprint(orderbook)

        price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((scalping_sell_amount)/price, 3)

        print("\n------------ Execute scalping sell -----------")
        print(f'{symbol} price : {price}, scalping sell amount = {amount}')  
        resp =exchange.create_market_sell_order(symbol=symbol, amount = amount )
        pprint(resp)

        logging.info(f"Scalping Sell order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def buy_coin(exchange,symbol: str)->None:
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol)
        pprint(orderbook)

        price = round(orderbook['asks'][0][0], 1)

        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (bb_trading_amount ):
            amount = (bb_trading_amount) 
        else:
            print("------- Cancel buy for low balance ------------")
            return

        print("\n------------ Make a buy order -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)
        pprint(resp)

        logging.info(f"Buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def scalping_buy_coin(exchange,symbol: str)->None:
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol)
        pprint(orderbook)

        price = round(orderbook['asks'][0][0], 1)

        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (scalping_buy_amount ):
            amount = (scalping_buy_amount) 
        else:
            print("------- Cancel buy for low balance ------------")
            return

        print("\n------------ Excute scalping buy -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)
        pprint(resp)

        logging.info(f"Scalping Buy order placed for {symbol} at price: {price}, amount = {amount}")

    except Exception as e:
        print("Exception : ", str(e))

def supertrend_sell_update(symbol: str):
    global supertrend_sell_amount
    supertrend_sell_amount[symbol] = supertrend_sell_quota[symbol]/ pow(1.5, supertrend_sell_iter[symbol])
    
    if supertrend_sell_amount[symbol] < 500000:
        supertrend_sell_amount[symbol] = 250000
 
def supertrend_sell_coin(exchange, symbol: str):
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol)
        pprint(orderbook)

        avg_price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        amount    = round((supertrend_sell_amount)/avg_price, 3)

        print("\n------------ Execute scalping sell -----------")
        print(f'{symbol} average price : {avg_price}, supertrend sell amount = {amount}')  
        resp =exchange.create_market_sell_order(symbol=symbol, amount = amount )
        pprint(resp)

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
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol)
        pprint(orderbook)

        free_KRW = exchange.fetchBalance()['KRW']['free']

        amount = 0.0
        if free_KRW > (supertrend_buy_amount ):
            amount = (supertrend_buy_amount) 
        else:
            print("------- Cancel buy for low balance ------------")
            return

        print("\n------------ Excute supertrend buy -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount = amount)
        pprint(resp)

    except Exception as e:
        print("Exception : ", str(e))

def execute_order(exchange, symbol: str)->None:
    
    sell   = sell_order[symbol]
    buy    = buy_order[symbol]

    if buy:
       buy_coin(exchange, symbol)
    elif sell:
       sell_coin(exchange, symbol)

    global iterations
    iterations[symbol] = iterations[symbol] + 1
    if (iterations[symbol] % 15 == 0):
       reset_sell_buy_order(symbol)

def execute_scalping_buy(exchange, symbol: str)->None:
    sell   = scalping_sell[symbol] 

    if sell:
       scalping_sell_coin(exchange, symbol)

def execute_scalping_sell(exchange, symbol: str)->None:
    buy    = scalping_buy[symbol]

    if buy:
        scalping_buy_coin(exchange, symbol)


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


def init_supertrend_quota(symbols):

    global supertrend_sell_quota 

    for x in symbols:
        supertrend_sell_quota[x]= 4000000

if __name__=='__main__':

    # Configure logging
    logging.basicConfig(filename="./trading.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    exchange = init_upbit()

    #define symbols 
    doge = "DOGE/KRW"
    xrp  = "XRP/KRW"
    sol  = "SOL/KRW"
    btc  = "BTC/KRW"

    symbols= [doge, xrp, sol, btc]
    init_supertrend_quota(symbols)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, doge)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, doge)
    schedule.every(1).minutes.do(execute_order, exchange, doge)
    schedule.every(3).minutes.do(execute_scalping_sell, exchange, doge)
    schedule.every(1).minutes.do(execute_scalping_buy, exchange, doge)
    schedule.every(4).hours.do(execute_supertrend_sell, exchange, doge)
    schedule.every(4).hours.do(execute_supertrend_buy, exchange, doge)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, xrp)

    schedule.every(1).minutes.do(execute_order, exchange, xrp)
    schedule.every(1).minutes.do(execute_scalping_buy, exchange, xrp)
    schedule.every(3).minutes.do(execute_scalping_sell, exchange, xrp)
    schedule.every(4).hours.do(execute_supertrend_sell, exchange, xrp)
    schedule.every(4).hours.do(execute_supertrend_buy, exchange, xrp)
    
    schedule.every(30).seconds.do(analyze_signals_1d, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, sol)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, sol)

    schedule.every(1).minutes.do(execute_order, exchange, sol)
    schedule.every(1).minutes.do(execute_scalping_buy, exchange, sol)
    schedule.every(3).minutes.do(execute_scalping_sell, exchange, sol)
    schedule.every(4).hours.do(execute_supertrend_sell, exchange, sol)
    schedule.every(4).hours.do(execute_supertrend_buy, exchange, sol)

    schedule.every(30).seconds.do(analyze_signals_1d, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, btc)
    schedule.every(30).seconds.do(analyze_supertrend, exchange, btc)

    schedule.every(1).minutes.do(execute_order, exchange, btc)
    schedule.every(1).minutes.do(execute_scalping_buy, exchange, btc)
    schedule.every(3).minutes.do(execute_scalping_sell, exchange, btc)
    schedule.every(4).hours.do(execute_supertrend_sell, exchange, btc)
    schedule.every(4).hours.do(execute_supertrend_buy, exchange, btc)

    schedule.every(30).seconds.do(monitor, symbols)
    schedule.every(30).seconds.do(monitor_balance, exchange)

    while True:
        schedule.run_pending()
        time.sleep(0.01)
