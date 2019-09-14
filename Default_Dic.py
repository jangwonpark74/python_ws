# defaultdict
# is particularly useful when making keying lists.

from collections import defaultdict

dd = defaultdict(list)
dd['one'].append('an item')
dd['one'].append('another item')
dd['two'].append('something else')

print(dd)
