import csv, random
import sys
import subprocess

HEADER = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'AIRLINE_ID', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

def addHeader(fname):
    with open(fname, "r") as fr:
        with open(fname + ".header", "w") as fw:
            reader = csv.reader(fr)
            writer(csv.writer(fw))
            writer.writerow(HEADER)
            for row in reader:
                writer.writerow(row)

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
