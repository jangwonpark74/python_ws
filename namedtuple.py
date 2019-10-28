from collections import namedtuple

Features = namedtuple('Features', ['age', 'gender', 'name'])
row = Features(age=22, gender='male', name='Alex')
print(row.age)
