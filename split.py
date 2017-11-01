import csv, random
import sys
import subprocess
from header import *

def inputMissing(fname):
    print (fname)
    with open(fname, "r") as fr:
        with open(fname + ".filled", "w") as fw:
            reader = csv.reader(fr)
            writer = csv.writer(fw)
            for row in reader:
                for i in ALL_DELAYS:
                    if row[i] == "":
                        row[i] = "0"
                writer.writerow(row)
    print ("Done " + fname)

inputMissing(sys.argv[1])


def shuffle(fname, N, k=5):
    # Shuffle a large csv file with N lines by performin a k-way
    # split, shuffling each part separately and randomly merging.

    # Define temporary csv files for split
    print("Define temporary csv files for split")
    fnames = ["tempCSV" + str(i) + ".csv" for i in range(k)]
    files = []
    writers = []
    for i in range(k):
        f = open(fnames[i], "w")
        files.append(f)
        writers.append(csv.writer(f))

    indices = list(range(k))

    # Split large file
    print("Split large file")
    with open(fname, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            i = random.choice(indices)
            writers[i].writerow(row)

    for f in files:
        f.close()

    # Shuffle small files
    print("Shuffle small files")
    for i in range(k):
        arr = []
        with open(fnames[i], "r") as f:
            reader = csv.reader(f)
            for row in reader:
                arr.append(row)
        print ("Shuffling file", i)
        random.shuffle(arr)
        with open(fnames[i], "w") as f:
            writer = csv.writer(f)
            for row in arr:
                writer.writerow(row)

    # Merge into large file
    print("Merge into large file")
    readers = [csv.reader(open(fnames[i], "r")) for i in range(k)]
    with open(fname + ".out", "w") as f:
        writer = csv.writer(f)
        activeReaders = k
        while activeReaders > 0:
            i = random.choice(indices)
            if readers[i] != None:
                row = next(readers[i], None)
                if row == None:
                    readers[i] = None
                    activeReaders -= 1
                else:
                    writer.writerow(row)
    for fn in fnames:
        subprocess.call(["rm", fn])

    print ("Done")

def spitTVT():
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
