def fn(**kwargs):
    for emp, age in kwargs.items():
        print("%s's age is %s" %(emp, age))

fn(John=25, Kalley=22, Tom=32)
