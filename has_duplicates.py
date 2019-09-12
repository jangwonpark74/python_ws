def has_duplicates(lst):
  return len(lst) != len(set(lst))
  
x = [1,2,3,4,5,5]
y = [1,2,3,4]

has_duplicates(x)
has_duplicates(y)
