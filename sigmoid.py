import math

def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s

val = basic_sigmoid(3)
print("value=" + str(val))
