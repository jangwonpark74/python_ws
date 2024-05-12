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

    markets = await exchange.load_markets()
    await exchange.close()
    return markets

if __name__ == '__main__':
    market =run(main())
    pprint(market['DOGE/USDT:USDT'])
