from collections import Counter

ages = [22, 22, 25, 25, 30, 24, 26, 24, 35, 45, 52, 20, 22, 22, 22]
value_counts = Counter(ages)
print(value_counts.most_common())
