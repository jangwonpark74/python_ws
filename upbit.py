import ccxt
import talib
import time
import schedule
import pandas as pd

from conf import key
from pprint import pprint
from datetime import datetime
from dataclasses import dataclass

from collections import defaultdict

# Bollinger band buy/sell order 
buy_order = defaultdict(bool)
sell_order = defaultdict(bool)

# MFI scalping 3 minute
scalping_buy = defaultdict(bool)
scalping_sell= defaultdict(bool)

# Global variable to keep the count for max 15 minutes continue for order
iterations = defaultdict(int)

# MFI 4hour for volatility analysis
mfi_4h = defaultdict(float)

# Free balance
free_balance = defaultdict(float)

# One minute buy or sell amount
one_minute_amount = 700000

# MFI and RSI based scalping 
scalping_amount = 5000000

#define for Currency dataclass
@dataclass(frozen=True)
class Currency:
    symbol:str

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

def reset_sell_buy_order(symbol):
    global sell_order
    global buy_order
    sell_order[symbol] = False
    buy_order[symbol] = False

def reset_sell_buy_order_4h(symbol):
    global sell_order_4h
    global buy_order_4h
    sell_order_4h[symbol] = False
    buy_order_4h[symbol] = False

def calc_volatility(x: float) -> float:
    volatility = round(-0.0012 * x * x + 0.12 * x + 0.5, 2)
    return volatility

def analyze_signals_3m(exchange, currency)->None:
    try:
        symbol = currency.symbol
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='3m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['volume'] = round(df['volume'], 1)
        df['mfi'] = round( talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 1)
        df['typical'] = round (( df['high'] + df['low'] + df['close'] ) / 3.0, 1)
        df['rsi'] = round( talib.RSI(df['close'], timeperiod=14), 1)


        # Scalping based on 3 minute MFI change
        mfi_3m = df['mfi'].iloc[-1]
        rsi_3m = df['rsi'].iloc[-1]

        global scalping_buy 
        global scalping_sell 
        scalping_sell[symbol] = ( mfi_3m > 80 ) 
        scalping_buy[symbol]  = ( mfi_3m < 20 ) 
        scalping_buy[symbol]  = scalping_buy_order[symbol] | ( rsi_3m < 30 ) 

        df['scaling_buy'] = scalping_sell[symbol]
        df['scaling_sell'] = scalping_buy[symbol]

    except Exception as e:
        print("Exception : ", str(e))

def analyze_signals_15m(exchange, currency)->None:
    try:
        symbol = currency.symbol
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['volume'] = round(df['volume'], 1)

        print(f'\n----------- {symbol} Bollinger Sell/Buy and Volatiltiy Analysis (15 minutes) --------------')
        # Calculate Bollinger Bands
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = talib.BBANDS(df['close'])
        df['bollinger_upper'] = round(df['bollinger_upper'], 1)
        df['bollinger_middle'] = round(df['bollinger_middle'], 1)
        df['bollinger_lower'] = round(df['bollinger_lower'], 1)
        df['mfi'] = round( talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 1)
        
        # update volatility 
        mfi = mfi_4h[symbol] 
        
        df['bollinger_width'] = round(((df['bollinger_upper'] - df['bollinger_lower'])/df['bollinger_middle']) * 100 , 3)
        bb_width = df['bollinger_width'].iloc[-1]

        global sell_order
        global buy_order

        bollinger_sell = (df['close'].iloc[-1] > df['bollinger_upper'].iloc[-1]) and (bb_width > calc_volatility(mfi)) 
        bollinger_buy = (df['low'].iloc[-1] < df['bollinger_lower'].iloc[-1]) and  (bb_width > calc_volatility(mfi))

        sell_order[symbol] = bollinger_sell
        buy_order[symbol] = bollinger_buy 

        df['bollinger_sell'] = bollinger_sell
        df['bollinger_buy'] = bollinger_buy

        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))


def analyze_signals_4h(exchange, currency)->None:
    try:
        symbol = currency.symbol
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        df['volume'] = round(df['volume'], 1)

        df['mfi'] = round( talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14), 1)

        global mfi_4h
        mfi_4h[symbol] = df['mfi'].iloc[-1]

        print(f'\n----------- {symbol} MFI analysis (4 hour) --------------')

        pprint(df.iloc[-1])

    except Exception as e:
        print("Exception : ", str(e))


def analyze_signals_1d(exchange, currency)->None:
    try:
        symbol = currency.symbol

        # 1 day bollinger analysis 
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1d')
        df_1d = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_1d['datetime'] = pd.to_datetime(df_1d['datetime'], utc=True, unit='ms')
        df_1d['datetime'] = df_1d['datetime'].dt.tz_convert("Asia/Seoul")
        df_1d['volume'] = round(df_1d['volume'], 1)

        # Calculate Bollinger Bands
        df_1d['bollinger_upper'], df_1d['bollinger_middle'], df_1d['bollinger_lower'] = talib.BBANDS(df_1d['close'])
        df_1d['bollinger_width'] = round(((df_1d['bollinger_upper'] - df_1d['bollinger_lower'])/df_1d['bollinger_middle']) * 100, 1)
        df_1d['bollinger_upper'] = round(df_1d['bollinger_upper'], 1)
        df_1d['bollinger_middle']= round(df_1d['bollinger_middle'], 1)
        df_1d['bollinger_lower'] = round(df_1d['bollinger_lower'], 1)
        df_1d['mfi']    = round( talib.MFI(df_1d['high'], df_1d['low'], df_1d['close'], df_1d['volume'], timeperiod=14), 1)

        print(f'\n----------------------- {symbol} Bolligner Band Analysis ( 1 day ) -----------------------------')
        pprint(df_1d.iloc[-1])

        # mfi crossover strategy
        mfi_now = round (df_1d['mfi'].iloc[-1], 2)

        # daily mfi
        global mfi
        mfi[symbol] = mfi_now

 
        print(f'\nSymbol: {symbol} mfi = {mfi_now}')

    except Exception as e:
        print("Exception : ", str(e))


def sell_coin(exchange, currency):
    try:
        symbol = currency.symbol
        orderbook = exchange.fetch_order_book(symbol)
        print("\n------------Getting order book -----------")
        pprint(orderbook)

        avg_price   = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        sell_amount = round((one_minute_amount)/ avg_price, 5)

        print("\n------------ Make a sell order-----------")
        print(f'{symbol} average price : {avg_price}, sell amount = {sell_amount}')  
        resp = exchange.create_market_sell_order(symbol=symbol, amount = sell_amount )
        pprint(resp)
    except Exception as e:
        print("Exception : ", str(e))

def scalping_sell_coin(exchange, currency):
    try:
        symbol = currency.symbol
        orderbook = exchange.fetch_order_book(symbol)
        print("\n------------Getting order book -----------")
        pprint(orderbook)

        avg_price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)

        scalping_sell_amount = round((scalping_amount)/ avg_price, 3)

        print("\n------------ Execute scalping sell -----------")
        print(f'{symbol} average price : {avg_price}, scalping sell amount = {scalping_sell_amount}')  
        resp =exchange.create_market_sell_order(symbol=symbol, amount = scalping_sell_amount )
        pprint(resp)

    except Exception as e:
        print("Exception : ", str(e))

def buy_coin(exchange,currency)->None:
    try:
        symbol = currency.symbol
        orderbook = exchange.fetch_order_book(symbol)
        print("\n------------Getting order book -----------")
        pprint(orderbook)

        avg_price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        buy_amount = 0
        if free_KRW > (one_minute_amount ):
            buy_amount = (one_minute_amount) 
        else:
            print("------- Cancel buy for low balance ------------")
            return

        print("\n------------ Make a buy order -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount=buy_amount)
        pprint(resp)

    except Exception as e:
        print("Exception : ", str(e))

def scalping_buy_coin(exchange,currency)->None:
    try:
        symbol = currency.symbol
        orderbook = exchange.fetch_order_book(symbol)
        print("\n------------Getting order book -----------")
        pprint(orderbook)

        avg_price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)
        free_KRW = exchange.fetchBalance()['KRW']['free']

        buy_amount = 0
        if free_KRW > (scalping_amount ):
            buy_amount = (scalping_amount) 
        else:
            print("------- Cancel buy for low balance ------------")
            return

        print("\n------------ Excute scalping buy -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount=scalping_amount)
        pprint(resp)

    except Exception as e:
        print("Exception : ", str(e))

def execute_order(exchange, currency)->None:

    symbol = currency.symbol

    # 15m analysis & buy/sell decision handling
    sell   = sell_order[symbol] 
    buy    = buy_order[symbol]

    if buy:
       buy_coin(exchange, currency)
    elif sell:
       sell_coin(exchange, currency)

    global iterations
    iterations[symbol] = iterations[symbol] + 1
    if (iterations[symbol] % 12== 0):
       reset_sell_buy_order(symbol)

def execute_scalping(exchange, currency)->None:
    
    symbol = currency.symbol

    # mfi based 1 minute sell/buy decision 
    sell   = scalping_sell[symbol] 
    buy    = scalping_buy[symbol]

    if buy:
       scalping_buy_coin(exchange, currency)
    elif sell:
       scalping_sell_coin(exchange, currency)

def monitor(x : list[Currency]):
    print("\n---------------- buy/sell order summary -----------------")

    column_name= ["Symbol", "Buy", "Sell", "Scalping Buy", "Scalping Sell"]
    orders = pd.DataFrame(columns = column_name)

    for y in x:
        s = y.symbol
        orders.loc[len(orders)] = [s, buy_order[s], sell_order[s], scalping_buy[s],scalping_sell[s]]
    pprint(orders)

if __name__=='__main__':

    exchange = init_upbit()

    # Define currency
    doge = Currency( symbol="DOGE/KRW")
    btc = Currency( symbol="BTC/KRW")
    xrp = Currency( symbol="XRP/KRW")
    eth = Currency( symbol="ETH/KRW")
    sol = Currency( symbol="SOL/KRW")

    #currencies = [doge, btc, xrp, eth, sol]
    currencies = [doge]
    
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, doge)
    schedule.every(1).minutes.do(execute_order, exchange, doge)
    schedule.every(3).minutes.do(execute_scalping, exchange, doge)
    
    """
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, xrp)
    schedule.every(1).minutes.do(execute_order, exchange, xrp)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, xrp)

    schedule.every(30).seconds.do(analyze_signals_3m, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, btc)
    schedule.every(1).minutes.do(execute_order, exchange, btc)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, btc)

    schedule.every(30).seconds.do(analyze_signals_3m, exchange, eth)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, eth)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, eth)
    schedule.every(1).minutes.do(execute_order, exchange, eth)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, eth)
    
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, sol)
    schedule.every(1).minutes.do(execute_order, exchange, sol)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, sol)
    """
    schedule.every(30).seconds.do(monitor, currencies)

    while True:
        schedule.run_pending()
        time.sleep(1)
