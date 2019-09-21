def logger(func):
    def wrapper():
        print('logging execution')
        func()
        print('Done logging')
    return wrapper

#decorator example
@logger
def sample():
    print('--- Inside sample function')

sample()
