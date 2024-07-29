import random
import numpy as np

def calc_pullback_price(symbol, price) -> float:
   r = abs(np.random.lognormal(0.03, 0.01) -1)
   return round(1-r, 3)

for i in range(30):
    print(i, "orgial price = 100", "pullback price =", calc_pullback_price("DOGE", 100))
