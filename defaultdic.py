from collections import defaultdict

my_default_dict = defaultdict(int)

for letter in 'the read fox ran as fast as it could':
	my_default_dict[letter] += 1

print(my_default_dict)
