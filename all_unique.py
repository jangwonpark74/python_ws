def all_unique(lst):
  return len(lst) == len(set(lst))
  
  
  x = [1, 1, 2, 3, 2, 3, 4, 5, 6]
  y = [1, 2, 3, 4, 5]
  
  print( x, "is unique ? = ", all_unique(x))
  print( y, "is unique ? = ", all_unique(y))
