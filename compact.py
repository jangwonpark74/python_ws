def compact(list):
  '''
     This method removes the false values from a list by using filter() method.
  '''
  return list(filter(bool, list))
