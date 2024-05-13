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

# Bollinger band criteria 15 minutes
buy_order = defaultdict(bool)
sell_order = defaultdict(bool)

# 3 minute MFI check 
check_mfi = defaultdict(float)

# 3 minute RSI check
check_rsi_3m = defaultdict(float)

# Bollinger band criteria 4h minutes
buy_order_4h = defaultdict(bool)
sell_order_4h = defaultdict(bool)

# mfi scalping 1 minute
scalping_buy_order = defaultdict(bool)
scalping_sell_order= defaultdict(bool)

# Pullback Buy
pullback_buy_order = defaultdict(bool)
pullback_high = defaultdict(float)
price = defaultdict(float)
pullback_buy_price = defaultdict(float)

# Global variable to keep the count for max 15 minutes continue for order
iterations = defaultdict(int)
iterations_4h = defaultdict(int)

mfi = defaultdict(float)

# Free balance
free_balance = defaultdict(float)

# One minute buy or sell amount
one_minute_amount = 700000

# MFI based scalping 
scalping_amount = 1000000


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
    global pullback_buy_order 
    sell_order[symbol] = False
    buy_order[symbol] = False
    pullback_buy_order[symbol] = False

def reset_sell_buy_order_4h(symbol):
    global sell_order_4h
    global buy_order_4h
    sell_order_4h[symbol] = False
    buy_order_4h[symbol] = False

def calc_volatility(x: float) -> float:
    volatility = round(-0.0012 * x * x + 0.12 * x + 0.5, 2)
    if volatility < 0:
        vlolatility = 0  
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
        global check_mfi
        check_mfi[symbol] = df['mfi'].iloc[-1]

        global check_rsi_3m
        check_rsi_3m[symbol] = df['rsi'].iloc[-1]


        global scalping_buy_order 
        global scalping_sell_order 

        if check_mfi[symbol] > 80 :
            scalping_sell_order[symbol] = True
        else:
            scalping_sell_order[symbol] = False

        if check_mfi[symbol] < 20 :
            scalping_buy_order[symbol] = True
        else:
            scalping_buy_order[symbol] = False

        if check_rsi_3m[symbol] < 30:
            scalping_buy_order[symbol] = scalping_buy_order[symbol] | True

        df['scaling_buy'] = scalping_sell_order[symbol]
        df['scaling_sell'] = scalping_buy_order[symbol]


        # Find high within 6hour   
        df['pullback_high'] = round (df['typical'].rolling(window = 72).max(), 1)

        global pullback_high
        global price
        pullback_high[symbol] = df['pullback_high'].iloc[-1]
        price[symbol] = df['typical'].iloc[-1]

        # pullback signal  
        pullback_threshold = 0.04
        df['pullback_price_check'] = df['close'] < ((1 - pullback_threshold) * df['pullback_high'])
        df['pullback_mfi_check'] = df['mfi'] < 30 

        pullback_buy = df['pullback_price_check'].iloc[-1] and df['pullback_mfi_check'].iloc[-1]
        df['pullback_buy'] = pullback_buy
        
        print(f'\n----------- {symbol} pullback analysis with mfi (3 minute) --------------')
        pprint(df.iloc[-1])

        global pullback_buy_order
        pullback_buy_order[symbol] = False
        pullback_buy_order[symbol] = pullback_buy_order[symbol] | pullback_buy 

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
        
        df['mfi_crossover_sell'] = False
        df['mfi_crossover_buy'] = False

        # update volatility 
        mfi_4h = mfi[symbol] 
        
        df['bollinger_width'] = round(((df['bollinger_upper'] - df['bollinger_lower'])/df['bollinger_middle']) * 100 , 3)
        bb_width = df['bollinger_width'].iloc[-1]

        volatility_threshold = calc_volatility(mfi_4h)
        bb_volatility_enough = False
        if bb_width >= volatility_threshold :
            bb_volatility_enough = True
        else:
            bb_volatility_enough = False

        global sell_order
        global buy_order

        bollinger_sell = (df['close'].iloc[-1] > df['bollinger_upper'].iloc[-1]) and bb_volatility_enough
        bollinger_buy = (df['low'].iloc[-1] < df['bollinger_lower'].iloc[-1]) and bb_volatility_enough

        sell_order[symbol] = bollinger_sell
        buy_order[symbol] = bollinger_buy 

        df['bollinger_sell'] = bollinger_sell
        df['bollinger_buy'] = bollinger_buy

        # mfi crossover strategy
        mfi_now = round (df['mfi'].iloc[-1], 2)
        mfi_last = round(df['mfi'].iloc[-2], 2)

        if (mfi_last >= 80) and (mfi_now < 80):
           mfi_sell_crossover = True 
        else:
           mfi_sell_crossover = False
    
        if (mfi_last <= 20) and (mfi_now >20):
            mfi_buy_crossover = True
        else:
            mfi_buy_crossover = False

        sell_order[symbol] = sell_order[symbol] |  mfi_sell_crossover
        buy_order[symbol] = buy_order[symbol] | mfi_buy_crossover

        df['mfi_crossover_sell'] = mfi_sell_crossover
        df['mfi_crossover_buy'] = mfi_buy_crossover
        pprint(df.iloc[-1])


    except Exception as e:
        print("Exception : ", str(e))

def analyze_signals_4h(exchange, currency)->None:
    try:
        symbol = currency.symbol

       # Investment Control Parameter Selection 
        print(f'\n----------------------- {symbol} MFI Analysis ( 4 hour ) -----------------------------')
        # Signal analysis one hour 
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe='4h')
        df_4h = pd.DataFrame(ohlcv_4h, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_4h['datetime'] = pd.to_datetime(df_4h['datetime'], utc=True, unit='ms')
        df_4h['datetime'] = df_4h['datetime'].dt.tz_convert("Asia/Seoul")
        df_4h['volume'] = round(df_4h['volume'], 1)

        # Caculated BB and Money Flow Index for trends
        df_4h['bollinger_upper'], df_4h['bollinger_middle'], df_4h['bollinger_lower'] = talib.BBANDS(df_4h['close'])

        df_4h['bollinger_sell'] = (df_4h['high'] > df_4h['bollinger_upper'])
        df_4h['bollinger_buy'] = (df_4h['low'] < df_4h['bollinger_lower'])
        df_4h['bollinger_width'] = round(((df_4h['bollinger_upper'] - df_4h['bollinger_lower'])/df_4h['bollinger_middle']) * 100, 3)
        df_4h['bollinger_upper'] = round(df_4h['bollinger_upper'], 1)
        df_4h['bollinger_middle']= round(df_4h['bollinger_middle'], 1)
        df_4h['bollinger_lower'] = round(df_4h['bollinger_lower'], 1)

        global sell_order_4h
        global buy_order_4h
        sell_order_4h[symbol] = df_4h['bollinger_sell'].iloc[-1]
        buy_order_4h[symbol] = df_4h['bollinger_buy'].iloc[-1]

        # 4h bollinger crossover signal detect 
        bb_4h_last = round(df_4h['bollinger_middle'].iloc[-2], 2) < df_4h['close'].iloc[-2]
        bb_4h_now =  round (df_4h['bollinger_middle'].iloc[-1], 2) > df_4h['close'].iloc[-1]

        bb_4h_sell_crossover = ( bb_4h_last == True  ) and ( bb_4h_now == True ) 
        bb_4h_buy_crossover  = ( bb_4h_last == False ) and ( bb_4h_now == False )

        df_4h['bollinger_crossover_sell'] = bb_4h_sell_crossover
        df_4h['bollinger_crossover_buy'] = bb_4h_buy_crossover

        sell_order_4h[symbol] = sell_order_4h[symbol] | bb_4h_sell_crossover 
        buy_order_4h[symbol] = buy_order_4h[symbol] | bb_4h_buy_crossover

        # mfi crossover strategy
        df_4h['mfi'] = round(talib.MFI(df_4h['high'], df_4h['low'], df_4h['close'], df_4h['volume'], timeperiod=14), 1)
        mfi_last = round(df_4h['mfi'].iloc[-2], 2)
        mfi_now = round (df_4h['mfi'].iloc[-1], 2)

        if (mfi_last >= 80) and (mfi_now < 80):
           mfi_crossover_sell = True 
        else:
           mfi_crossover_sell = False
    
        if (mfi_last <= 20) and (mfi_now >20):
            mfi_crossover_buy = True
        else:
            mfi_crossover_buy = False
        
        sell_order_4h[symbol] = sell_order_4h[symbol] | mfi_crossover_sell
        buy_order_4h[symbol] = buy_order_4h[symbol] | mfi_crossover_buy
        
        df_4h['mfi_crossover_sell'] = mfi_crossover_sell 
        df_4h['mfi_crossover_buy'] = mfi_crossover_buy 

        pprint(df_4h.iloc[-1])

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

        avg_price = round((orderbook['bids'][0][0] + orderbook['asks'][0][0])/2, 1)

        sell_amount = round((one_minute_amount)/ avg_price, 5)

        if check_mfi[symbol] < 35 :
            print(f'{symbol} Cancel sell for low one minute mfi = {check_mfi[symbol]}')  
            return

        print("\n------------ Make a sell order-----------")
        print(f'{symbol} average price : {avg_price}, sell amount = {sell_amount}')  
        resp =exchange.create_market_sell_order(symbol=symbol, amount = sell_amount )
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

        print("\n------------ Make a scalping sell order-----------")
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

        if check_mfi[symbol] > 75 :
            print(f'{symbol} Cancel buy for high one minute mfi = {check_mfi[symbol]}')  
            return

        buy_amount = 0
        if free_KRW > (one_minute_amount ):
            buy_amount = (one_minute_amount) 
        else:
            print("------- Cancel buy for low balance ------------")
            return

        global pullback_buy_price

        if pullback_buy_order[symbol] == True:
            pullback_buy_price[symbol] = avg_price

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

        print("\n------------ Make a scalping buy order -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = symbol, amount=scalping_amount)
        pprint(resp)

    except Exception as e:
        print("Exception : ", str(e))

def execute_order(exchange, currency)->None:
    global iterations

    symbol = currency.symbol

    # 15m analysis & buy/sell decision handling
    sell   = sell_order[symbol] 
    buy    = buy_order[symbol]

    if buy:
       buy_coin(exchange, currency)
    elif sell:
       sell_coin(exchange, currency)

    buy = pullback_buy_order[symbol]
    if buy:
       buy_coin(exchange, currency)

    iterations[symbol] = iterations[symbol] + 1
    if (iterations[symbol] % 12== 0):
       reset_sell_buy_order(symbol)

def execute_scalping_order(exchange, currency)->None:
    global iterations

    symbol = currency.symbol

    # mfi based 1 minute sell/buy decision 
    sell   = scalping_sell_order[symbol] 
    buy    = scalping_buy_order[symbol]

    if buy:
       scalping_buy_coin(exchange, currency)
    elif sell:
       scalping_sell_coin(exchange, currency)

def execute_order_4h(exchange, currency)->None:

    symbol = currency.symbol

    # 4h analysis & buy/sell decision handling
    sell   = sell_order_4h[symbol] 
    buy    = buy_order_4h[symbol]

    if buy:
       buy_coin(exchange, currency)
    elif sell:
       sell_coin(exchange, currency)

    global iterations_4h
    iterations_4h[symbol] = iterations_4h[symbol] + 1
    if (iterations_4h[symbol] % 16== 0):
       reset_sell_buy_order_4h(symbol)

@dataclass(frozen=True)
class Currency:
    symbol:str

def monitor_buy_sell_order(x : list[Currency]):
    print("\n---------------- buy/sell order summary -----------------")

    column_name= ["Symbol", "Buy", "Sell", "Pullback", "Scalping Buy", "Scalping Sell"]
    orders = pd.DataFrame(columns = column_name)

    for y in x:
        s = y.symbol
        orders.loc[len(orders)] = [s, buy_order[s], sell_order[s], pullback_buy_order[s], scalping_buy_order[s],scalping_sell_order[s]]
    pprint(orders)

def fetch_balance(exchange, x: list[Currency]):

    try:
        free_DOGE = exchange.fetch_balance()['DOGE']['free']

        print("\n--------------- free balance  -------------------------")
        print(f"Free Doge ={free_DOGE}")
    
    except Exception as e:
        print("Exception : ", str(e))



if __name__=='__main__':

    exchange = init_upbit()

    # Define currency
    doge = Currency( symbol="DOGE/KRW")
    btc = Currency( symbol="BTC/KRW")
    xrp = Currency( symbol="XRP/KRW")
    eth = Currency( symbol="ETH/KRW")
    sol = Currency( symbol="SOL/KRW")

    #currencies = [doge, btc, xrp, eth, sol]
    currencies = [doge, xrp]
    
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, doge)
    schedule.every(1).minutes.do(execute_order, exchange, doge)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, doge)
    schedule.every(15).minutes.do(execute_order_4h, exchange, doge)
    
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, xrp)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, xrp)
    schedule.every(1).minutes.do(execute_order, exchange, xrp)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, xrp)
    schedule.every(15).minutes.do(execute_order_4h, exchange, xrp)

    """
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, btc)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, btc)
    schedule.every(1).minutes.do(execute_order, exchange, btc)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, btc)
    schedule.every(15).minutes.do(execute_order_4h, exchange, btc)

    schedule.every(30).seconds.do(analyze_signals_3m, exchange, eth)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, eth)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, eth)
    schedule.every(1).minutes.do(execute_order, exchange, eth)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, eth)
    schedule.every(15).minutes.do(execute_order_4h, exchange, eth)
    
    schedule.every(30).seconds.do(analyze_signals_3m, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_4h, exchange, sol)
    schedule.every(30).seconds.do(analyze_signals_15m, exchange, sol)
    schedule.every(1).minutes.do(execute_order, exchange, sol)
    schedule.every(3).minutes.do(execute_scalping_order, exchange, sol)
    schedule.every(15).minutes.do(execute_order_4h, exchange, sol)
    """
    schedule.every(30).seconds.do(monitor_buy_sell_order, currencies)
    schedule.every(30).seconds.do(fetch_balance, exchange, currencies)

    while True:
        schedule.run_pending()
        time.sleep(1)
