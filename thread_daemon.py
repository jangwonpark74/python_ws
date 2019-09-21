import threading

def print_something(data):
    print(data)

# creating daemon thread 
t = threading.Thread(target=print_something, args=("hello",), daemon=True)
t.start()
print("thread started")
