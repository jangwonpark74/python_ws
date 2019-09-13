import math

import numpy as np 
import matplotlib.pyplot as plt

from sklearn import linear_model

x = np.random.rand(1000,1)
x = x *20 -10

y = np.array([math.sin(v) for v in x])
y += np.random.randn(1000)

model = linear_model.LinearRegression()
model.fit( x, y)

print(model.score(x,y))

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x), marker='o')
plt.show()
