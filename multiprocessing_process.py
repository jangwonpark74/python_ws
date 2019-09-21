import random
from multiprocessing import Manager, Process
# no file should have name multiprocessing when import multiprocessing

def compute(results):
    results.append(sum([random.randint(1, 100) for i in range(1000000)]))

if __name__ == '__main__':
    with Manager() as manager:
        results = manager.list()
        workers = [Process(target=compute, args=(results,)) for i in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        print("Results : %s" % results
