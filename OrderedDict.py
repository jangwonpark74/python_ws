# ordered dictionary preserves the order by 
# maintaining a list of keys inserted into the dictionary
# in the order in which they have been inserted
# The order is also preserved withen deleting keys from an OrderedDict. 

from collections import OrderedDict

od = OrderedDict()
od['key1'] = 1
od['key2'] = 'a'
od['key3'] = 'alpha'

plain = dict(od)
print(list(od.keys()))

del od['key1']
print(list(od.keys()))
