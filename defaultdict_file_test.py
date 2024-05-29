import pickle
from collections import defaultdict

def save_data(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Example usage
filename = 'data_store.pkl'

# Create or load defaultdict
try:
    data = load_data(filename)
except FileNotFoundError:
    data = defaultdict(list)

# Modify data
data['key1'].append(1)
data['key2'].extend([2, 3, 4])

# Save data
save_data(filename, data)

# Load and print data
loaded_data = load_data(filename)
print("Loaded data:", loaded_data)

# again 
data['key1'].append(10)
data['key2'].extend([20, 30, 40])

save_data(filename, data)

# Load and print data
loaded_data = load_data(filename)
print("2nd Loaded data:", loaded_data)

