import time
import os
import sys
import ccxt
import pandas as pd
import numpy as np
import schedule
from schedule import every, repeat
import datetime

from conf import key

n = 14  # RSI period
symbol = "DOGE/KRW"

def init_upbit():
    print('CCXT Version:', ccxt.__version__)

    exchange = ccxt.upbit(config={
            'apiKey':key['accessKey'],
            'secret':key['secret'],
            'enableRateLimit': True
        }
    )

    return exchange

def get_doge_price(exchange):
    doge_infos = exchange.fetch_ticker(symbol)
    doge_price = doge_infos["close"]
    print(f'current doge_price = {doge_price}')

def get_doge_ohlcv(exchange, duration):
    doge_ohlcv = exchange.fetch_ohlcv(symbol=symbol, timeframe=duration)
    return doge_ohlcv

def get_dataframe(ohlcv):
    df = pd.DataFrame(ohlcv, columns =['datetime', 'open', 'high', 'low', 'close', 'volume'])
    pd_ts = pd.to_datetime(df['datetime'], utc=True, unit='ms')     # unix timestamp to pandas Timeestamp
    pd_ts = pd_ts.dt.tz_convert("Asia/Seoul")                       # convert timezone
    pd_ts = pd_ts.dt.tz_localize(None)
    df.set_index(pd_ts, inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    print(df.tail(120))

def df_seoul(ohlcv):
    df = pd.DataFrame(ohlcv, columns =['datetime', 'open', 'high', 'low', 'close', 'volume'])
    pd_ts = pd.to_datetime(df['datetime'], utc=True, unit='ms')     # unix timestamp to pandas Timeestamp
    pd_ts = pd_ts.dt.tz_convert("Asia/Seoul")                       # convert timezone
    pd_ts = pd_ts.dt.tz_localize(None)
    df.set_index(pd_ts, inplace=True)
    return df

def calculate_bollinger_with_rsi(df):
    df = df[['open', 'high', 'low', 'close', 'volume', 'rsi']]
    df['middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(20).std(ddof=0)
    df['upper'] = df['middle'] + 2 * std
    df['upper'] = df['upper'].apply(lambda x: round(x, 1))
    df['lower'] = df['middle'] - 2 * std
    df['lower'] = df['lower'].apply(lambda x: round(x, 1))
    df['middle']= df['middle'].apply(lambda x: round(x, 1))
    return df

def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

def RSI(ohlcv):
    df = df_seoul(ohlcv)

    # RSI calculation logic
    df['change'] =  df['close'].diff() # 종가 차이 계산
    df['gain']= df.change.mask(df.change <0, 0.0)
    df['loss']= -df.change.mask(df.change>0, -0.0)
    df['avg_gain'] = rma(df.gain[n+1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n+1])/n)

    df['change'] = df['close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df['avg_gain'] = rma(df.gain[n+1:].to_numpy(), n, np.nansum(df.gain.to_numpy()[:n+1])/n)
    df['avg_loss'] = rma(df.loss[n+1:].to_numpy(), n, np.nansum(df.loss.to_numpy()[:n+1])/n)
    df['rs'] = df.avg_gain / df.avg_loss
    df['rsi'] = 100 - (100 / (1 + df.rs))
    df['rsi'] = df['rsi'].apply(lambda x: round(x, 1))
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'rsi']]
    return df

def bollinger_rsi(ohlcv):
    df = RSI(ohlcv)
    df = calculate_bollinger_with_rsi(df)
    return df

def sell_ohlcv(ohlcv_rs):
    df = ohlcv_rs
    condition1 = df['rsi'] > 70
    condition2 = df['close'] > df['upper']
    df_sell= df.loc[condition1 & condition2]
    return df_sell

def buy_ohlcv(ohlcv_rs):
    df = ohlcv_rs
    condition3 = df['rsi'] < 30
    condition4 = df['close'] < df['lower']
    df_buy = df.loc[condition3 & condition4]
    return df_buy

def fetch_balance(exchange):
    balance = exchange.fetch_balance()
    info = balance['info']
    df = pd.DataFrame.from_dict(info)

    df['symbol'] = df['currency'] +'/' + df['unit_currency']
    df = df.drop(columns=['currency', 'avg_buy_price_modified', 'unit_currency'])
    df['total_volume'] = pd.to_numeric(df['balance']) + pd.to_numeric(df['locked'])
    df = df[['symbol', 'balance', 'locked', 'avg_buy_price', 'total_volume']]
    df = df.rename(columns={'balance': 'free_volume', 'locked': 'ordered_volume', 'avg_buy_price': 'purchase_price'})
    df['index'] = df.index
    df.loc[df['total_volume']>0, 'open'] = True
    symbol_list = df['symbol'].values.tolist()
    print(df)
    return symbol_list, df

def market_sell_order(exchange, symbol, amount):

    print("--------------market sell order -----------")
    ticker = exchange.fetch_ticker(symbol)
    price = ticker["close"]
    print(datetime.datetime.now())
    print(symbol, " price: ", price, "amount: ", amount, "total: ", (price *amount), "KRW")
#    resp = exchange.create_market_sell_order( symbol=symbol, amount=amount)
#    print(resp)

def limit_buy_order(exchange, symbol, amount):
    ticker = exchange.fetch_ticker(symbol)
    price = ticker["close"]
    price = round((price * 0.99), 2)
    print("\n------------limit_buy_order-----------")
    print("symbol :", symbol)
    print("price  :", price, " amount: ", amount, "total :", (price * amount), "KRW")
#    resp = exchange.create_limit_buy_order(symbol = symbol, amount = amount, price = price)
#    print(resp)


if __name__=='__main__':
    exchange = init_upbit()
    doge_ohlcv = get_doge_ohlcv(exchange, '15m')
    doge_ohlcv_br = bollinger_rsi(doge_ohlcv)

    print("--------sell---------")
    print(sell_ohlcv(doge_ohlcv_br))

    print("\n")
    print("--------buy---------")
    print(buy_ohlcv(doge_ohlcv_br))

    print("\n--------balance---------")
    fetch_balance(exchange)

    print("\n--------doge price---------")
    get_doge_price(exchange)

    limit_buy_order(exchange, symbol, 1000)
    market_sell_order(exchange, symbol, 1000)
