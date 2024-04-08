import os
import time
import sys
import schedule
import ccxt
import talib
import pandas as pd

from conf import key
from pprint import pprint
from datetime import datetime
from dataclasses import dataclass

buy_order = False
sell_order = False
avg_buy_price = 0.0
account_money = 0.0
profit_factor = 1.12
btc_sell_amount = 0.02
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

def reset_sell_buy_order():
    global sell_order
    global buy_order
    sell_order = False
    buy_order = False

def analyze_signals(exchange, currency)->None:
    try:
        ohlcv = exchange.fetch_ohlcv(symbol=currency.symbol, timeframe='15m')
        df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, unit='ms')
        df['datetime'] = df['datetime'].dt.tz_convert("Asia/Seoul")
        
       # Calculate Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'])
        df['bollinger_pb'] = (df['close'] - df['lower_band'])/(df['upper_band'] - df['lower_band']) * 100
        df['bollinger_sell'] = (df['high'] > df['upper_band']) | (df['open'] > df['upper_band']) | \
                                (df['bollinger_pb'] > 100)

        df['bollinger_buy'] = (df['low'] < df['lower_band']) | (df['open'] < df['lower_band']) | \
                              (df['bollinger_pb'] < 0)
        df['price_judge']  = False
        df['sell_price']  = avg_buy_price * profit_factor
        df['price_judge'] = (df['high'] > df['sell_price'])

        global sell_order
        sell_order = df['bollinger_sell'].iloc[-1] & df['price_judge'].iloc[-1]

        global buy_order
        buy_order = df['bollinger_buy'].iloc[-1]
        print("\n----------------------- Buy or Sell ------------------------------")
        print( "sell_order : " , sell_order, "  ", "buy_order : ", buy_order)

        print("\n---------------------signal analysis-----------------------------")
        pprint(df.iloc[-1])
    except Exception as e:
        print("Exception : ", str(e))

def get_balance(exchange, currency) -> None:
    try:
        balance = exchange.fetch_balance()
        df = pd.DataFrame.from_dict(balance['info'])
        df['symbol'] = df['currency'] +'/' + df['unit_currency']
        df = df.drop(columns=['currency', 'avg_buy_price_modified', 'unit_currency'])
        df['total_volume'] = pd.to_numeric(df['balance']) + pd.to_numeric(df['locked'])
        df = df[['symbol', 'balance', 'locked', 'avg_buy_price', 'total_volume']]
        df = df.rename(columns={'balance': 'free_volume', 'locked': 'ordered_volume', 'avg_buy_price': 'purchase_price'})
        df['index'] = df.index
        money_df = df.loc[df['symbol'] == "KRW/KRW"]
        x = money_df['free_volume'].values

        global account_money
        account_money = float(x[0])

        global avg_buy_price
        symbol_df = df.loc[df['symbol'] == currency.symbol]
        y = symbol_df['purchase_price'].iloc[-1]
        avg_buy_price = float(y)

        print("\n------------------- My Balance --------------------------------")
        pprint(df)
        
  except Exception as e:
        print("Exception : ", str(e))

def get_order_book(exchange, currency):
    try:
        orderbook = exchange.fetch_order_book(symbol=currency.symbol)
        print("\n-------------Get order book of {} before BUY -----------", currency.symbol)
        pprint(orderbook)
    except Exception as e:
        print("Exception : ", str(e))

def sell_coin(exchange, currency):
    try:
        print("\n-------------market sell order -----------")
        print("Sell ", currency.symbol)
        get_balance(exchange, currency)
        resp =exchange.create_market_sell_order(symbol=currency.symbol, amount = currency.one_minute_sell_amount)
        pprint(resp)
    except Exception as e:
        print("Exception : ", str(e))


def buy_coin(exchange,currency)->None:
    try:
        print("\n------------Make a Buy -----------")
        get_order_book(exchange, currency)
        exchange.options['createMarketBuyOrderRequiresPrice']=False
        resp = exchange.create_market_buy_order(symbol = currency.symbol, amount=currency.one_minute_buy_quota)
        pprint(resp)
    except Exception as e:
        print("Exception : ", str(e))

def execute_order(exchange, currency)->None:
    global sell_order
    global buy_order
    global iterations

    if sell_order:
       sell_coin(exchange, currency)
    if buy_order:
       buy_coin(exchange, currency)

    iterations += 1
    if (iterations % 15 == 0):
       reset_sell_buy_order()

def fill_account_money(exchange, currency):
    if account_money < currency.refill_amount:
        try:
            resp = exchange.create_market_sell_order(symbol = currency.symbol, amount = btc_sell_amount)
            pprint(resp)
        except Exception as e:
            print("Exception : ", str(e))

@dataclass(frozen=True)
class Currency:
    symbol:str
    one_minute_buy_quota:float
    one_minute_sell_amount:float
    refill_amount:float

if __name__=='__main__':
    exchange = init_upbit()

    doge = Currency( symbol="DOGE/KRW",
                     one_minute_buy_quota = 80000,
                     one_minute_sell_amount = 350,
                     refill_amount = 1500000
                   )

    btc  = Currency( symbol="BTC/KRW",
                     one_minute_buy_quota = 1000000,
                     one_minute_sell_amount = 0.003,
                     refill_amount=1500000
                   )
    schedule.every(15).seconds.do(get_balance, exchange, doge)
    schedule.every(30).seconds.do(analyze_signals, exchange, doge)
    schedule.every(1).minutes.do(execute_order, exchange, doge)
    schedule.every(1).minutes.do(fill_account_money, exchange, btc)

    while True:
        schedule.run_pending()
        time.sleep(1)
