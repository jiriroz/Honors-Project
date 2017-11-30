import csv
from header import *

# Split by airport and sort each one based on flight number and date.

nWrite = 10000

def appendToCsv(fname, rows):
    with open(fname, "a+") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def getFlNum(row):
    return row[UNIQUE_CARRIER].strip() + row[FL_NUM].strip()

def getAirlineKey(row):
    return "{}-{}".format(row[UNIQUE_CARRIER].strip(), row[YEAR].strip())

def getAirportKey(row):
    return "{}".format(row[ORIGIN_AIRPORT_ID].strip())

fname = "data/test.csv"

with open(fname, "r") as fr:
    reader = csv.reader(fr)
    i = -1
    data = dict()
    header = []
    for row in reader:
        if i == -1:
            header = row
            i = 0
            continue
        i += 1
        airport = getAirportKey(row)
        if airport not in data:
            data[airport] = []
        data[airport].append(row)
        if len(data[airport]) > nWrite:
            print ("Writing to", airport)
            appendToCsv("temp/{}.csv".format(airport), data[airport])
            data[airport] = []
    for airport in data:
        print ("Writing to", airport)
        appendToCsv("temp/{}.csv".format(airport), data[airport])
        data[airport] = []

    with open(fname + ".sorted", "w") as fw:
        writer = csv.writer(fw)
        writer.writerow(header)
        for airport in sorted(list(data.keys())):
            print ("Processing", airport)
            rows = []
            with open("temp/{}.csv".format(airport), "r") as fr:
                reader = csv.reader(fr)
                for row in reader:
                    rows.append(row)
                rows.sort(key = lambda x: (getFlNum(x), x[0], x[1], x[2]))
            for row in rows:
                writer.writerow(row)
    print ("Done")
            
        
