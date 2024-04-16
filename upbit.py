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

buy_order = defaultdict(bool)
sell_order = defaultdict(bool)
iterations =0

def init_upbit():
    print('\n-----------------Upbit Exchange Initialization-------------------------')
    print('Initialized CCXT with version : ', ccxt.__version__)
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

def analyze_signals(exchange, currency)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol=currency.symbol, timeframe='15m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")

        ohlcv_60 = exchange.fetch_ohlcv(symbol=currency.symbol, timeframe='1h')
        df_60 = pd.DataFrame(ohlcv_60, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df_60['datetime'] = pd.to_datetime(df_60['datetime'], utc=True, unit='ms')
        df_60['datetime'] = df_60['datetime'].dt.tz_convert("Asia/Seoul")

        # Calculate Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'])

        # Calculate Bollinger Bands 60m
        df_60['upper_band'], df_60['middle_band'], df_60['lower_band'] = talib.BBANDS(df_60['close'])

        print("\n----------------------- bollinger_1h analysis ---------------------------")
        pprint(df_60.iloc[-1])

        print("\n----------------------- Trend analysis ---------------------------")
        uptrend = df['close'].iloc[-1] > df_60['middle_band'].iloc[-1]
        downtrend = df['close'].iloc[-1] < df_60['middle_band'].iloc[-1]
        print("Uptrend: ", uptrend, "Downtrend: ", downtrend)

        df['bollinger_sell'] = (df['high'] > df['upper_band'])
        df['bollinger_buy'] = (df['low'] < df['lower_band'])

        print("\n---------------------signal analysis-----------------------------")
        pprint(df.iloc[-1])

        symbol = currency.symbol

        global sell_order
        global buy_order
        sell_order[symbol] = df['bollinger_sell'].iloc[-1]
        buy_order[symbol] = df['bollinger_buy'].iloc[-1]
        print("\n----------------------- Buy or Sell ------------------------------")
        print( "Symbol:", symbol,"sell_order : " , sell_order[symbol], "  ", "buy_order : ", buy_order[symbol])

    except Exception as e:
        print("Exception : ", str(e))

def sell_coin(exchange, currency):
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol=currency.symbol)
        pprint(orderbook)

        resp =exchange.create_market_sell_order(symbol=currency.symbol, amount = currency.one_minute_sell_amount)
        pprint(resp)
    except Exception as e:
        print("Exception : ", str(e))

def buy_coin(exchange,currency)->None:
    try:
        print("\n------------Getting order book -----------")
        orderbook = exchange.fetch_order_book(symbol=currency.symbol)
        pprint(orderbook)

        one_minute_buy_quota = 300000
        free_KRW = exchange.fetchBalance()['KRW']['free']
        if free_KRW > one_minute_buy_quota:
            amount = one_minute_buy_quota
        else:
            print("------- cancel buy for low Balance ------------")
            return

        print("\n------------ make a Buy -----------")
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = currency.symbol, amount=amount)
        pprint(resp)

    except Exception as e:
        print("Exception : ", str(e))

def execute_order(exchange, currency)->None:
    global iterations

    sell = sell_order[currency.symbol]
    buy = buy_order[currency.symbol]

    if sell:
       sell_coin(exchange, currency)
    if buy:
       buy_coin(exchange, currency)

    iterations += 1
    if (iterations % 15 == 0):
       reset_sell_buy_order(currency.symbol)

@dataclass(frozen=True)
class Currency:
    symbol:str
    one_minute_sell_amount:float

if __name__=='__main__':
    exchange = init_upbit()

    doge = Currency( symbol="DOGE/KRW", one_minute_sell_amount = 1000)
    btc = Currency( symbol="BTC/KRW", one_minute_sell_amount = 0.0026)
    xrp = Currency( symbol="XRP/KRW", one_minute_sell_amount = 400)

    schedule.every(30).seconds.do(analyze_signals, exchange, doge)
    schedule.every(1).minutes.do(execute_order, exchange, doge)
    schedule.every(31).seconds.do(analyze_signals, exchange, xrp)
    schedule.every(1).minutes.do(execute_order, exchange, xrp)

    while True:
        schedule.run_pending()
        time.sleep(1)
