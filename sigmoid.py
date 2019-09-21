import math
import numpy as np 

def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s

val = basic_sigmoid(3)
print("value=" + str(val))

x = [1, 2, 3, 4]
print("x= " + str(x))
y = np.array(x)
print("y= " +str(y))

val = 1/(1+ np.exp(x))
print("val=" + str(val))
