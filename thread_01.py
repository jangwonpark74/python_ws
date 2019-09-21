import threading

def print_something(data):
    print(data)

# take care when define args
t = threading.Thread(target=print_something, args=("hello",))
t.start()
print("thread started")
#join make it deterministic to finish main thread to terminated 
t.join()
