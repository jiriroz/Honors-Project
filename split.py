import csv, random
import sys

data = list(range(34919683))
random.shuffle(data)

N = 5237952
val = set(data[:N])
test = set(data[N:2*N])
train = set(data[2*N:])

print (val & test)
print (train & val)
print (len(val), len(test), len(train))

valCt, testCt, trainCt = 0, 0, 0
valW = csv.writer(open("_/val.csv", "w"))
testW = csv.writer(open("_/test.csv", "w"))
trainW = csv.writer(open("_/train.csv", "w"))

limit = 10000000000000000

zeroes = 0
i = -1

with open("data/all.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        _ = -1
        if i == -1:
            i = 0
            continue
        if i in val:
            valCt += 1
            _ = valW.writerow(row)
        elif i in test:
            testCt += 1
            _ = testW.writerow(row)
        else:
            trainCt += 1
            _ = trainW.writerow(row)
        if _ == 0:
            zeroes += 1
        i += 1
        if i >= limit:
            break

print (valCt, testCt, trainCt)
print (zeroes)
