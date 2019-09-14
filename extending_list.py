def extend_list(val, list=None):
    if list is None:
        list = []
    list.append(val)
    return list

list1 = extend_list(10)
list2 = extend_list(123, [])
list3 = extend_list('a')

print('list1=', list1)
print('list2=', list2)
print('list3=', list3)
