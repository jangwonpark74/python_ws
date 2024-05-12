import os
import time
import sys
import schedule
import asyncio
import ccxt.async_support as ccxt
import talib
import pandas as pd

from conf import binance_key
from pprint import pprint
from datetime import datetime

from asyncio import run



def table(values):
    first = values[0]
    keys = list(first.keys()) if isinstance(first, dict) else range(0, len(first))
    widths = [max([len(str(v[k])) for v in values]) for k in keys]
    string = ' | '.join(['{:<' + str(w) + '}' for w in widths])
    return "\n".join([string.format(*[str(v[k]) for k in keys]) for v in values])


async def main():
    print('\n-----------------Binance Exchange Initialization-------------------------')
    print('Initialized CCXT with version : ', ccxt.__version__)
    
    exchange = ccxt.binanceusdm(config={
            'apiKey':binance_key['apiKey'],
            'secret':binance_key['secret'],
            'enableRateLimit': True,
    })

    since = exchange.parse8601('2024-04-28T00:00:00Z')
    symbol = 'DOGE/USDT'
    timeframe = '15m'
    count = 0
    all_ohlcvs = []

    while True:
        try:
            ohlcvs = await exchange.fapiPublic_get_continuousKlines(params)
            print(table([o for o in ohlcvs]))
            print(table([[exchange.iso8601(int(o[0]))] + o[1:] for o in ohlcvs]))
        except Exception as e:
            print(e, str(e))
        await exchange.close()
                     
    df = pd.DataFrame(all_ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], utc=True, unit='ms')
    df = df.drop(['timestamp'], axis=1)
    df['date'] = df['date'].dt.tz_convert("Asia/Seoul")
    df.to_csv('Dogehistory_15m.csv', index=False)

if __name__ == '__main__':
    run(main())
