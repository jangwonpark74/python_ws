import os
import time
import sys
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from conf import key
from pprint import pprint
from datetime import datetime
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Currency:
    symbol:str

if __name__=='__main__':
    exchange = init_upbit()
    portfolio = ["DOGE/KRW", "BTC/KRW", "XRP/KRW", "ETH/KRW", "SOL/KRW"] 
    analyze_covariance(exchange, portfolio)

