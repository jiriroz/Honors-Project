import csv

# Split by airline and sort each one based on flight number and date.

nWrite = 10000

def appendToCsv(fname, rows):
    with open(fname, "a+") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

def getFlNum(row):
    return row[4].strip() + row[7].strip()

def getAirline(row):
    return row[4].strip()


with open("data/2016.csv", "r") as fr:
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
        if i % 100000 == 0:
            print ( i / 63490.0, "%")
        airline = getAirline(row)
        if airline not in data:
            data[airline] = []
        data[airline].append(row)
        if len(data[airline]) > nWrite:
            print ("Writing to", airline)
            appendToCsv("temp/{}.csv".format(airline), data[airline])
            data[airline] = []
    for airline in data:
        print ("Writing to", airline)
        appendToCsv("temp/{}.csv".format(airline), data[airline])
        data[airline] = []

    with open("data/2016-sorted.csv", "w") as fw:
        writer = csv.writer(fw)
        writer.writerow(header)
        for airline in sorted(list(data.keys())):
            print ("Processing", airline)
            rows = []
            with open("temp/{}.csv".format(airline), "r") as fr:
                reader = csv.reader(fr)
                for row in reader:
                    rows.append(row)
                rows.sort(key = lambda x: (getFlNum(x), x[1], x[2]))
            for row in rows:
                writer.writerow(row)
            
        


    #print ("Sorting")
    #for flNum in data:
    #    rows = data[flNum]
    #    rows.sort(key = lambda x: (x[1], x[2]))
    #print ("Sorted")
    #print ("Sample: ")
    #for flNum in list(data.keys())[:10]:
    #    for x in data[flNum][:7]:
    #        print (x[1], x[2], x[4], x[7])
    #    print ()
