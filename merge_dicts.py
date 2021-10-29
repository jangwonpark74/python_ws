def merge_dicts(a, b):
  c = a.copy()        # make a copy of a
  c.update(b)         # modify map a with b
  return c

a = {'x':1, 'y': 2}
b = {'y':3, 'z' : 4}
print(a, b)
print(merge_dicts(a,b))
