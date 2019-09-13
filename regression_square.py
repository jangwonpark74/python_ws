import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

x = np.random.rand(100,1)

x = x * 4 -2
y = 3 * x**2 - 2

y += np.random.randn(100,1)

model = linear_model.LinearRegression()
model.fit(x**2, y)

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x**2), marker='o')
plt.show()

print( model.coef_)
print( model.intercept_)

print( model.score(x**2,y))
