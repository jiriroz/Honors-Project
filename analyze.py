import csv
import os
import numpy as np
import matplotlib.pyplot as plt

HEADER = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'AIRLINE_ID', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', '']

def main():
    datadir = "data/"
    datafiles = [x for x in os.listdir(datadir) if x[0] != "." and x[-3:] == "csv"]

    totalct = 0
    missing = [0] * len(HEADER)

    for fname in datafiles:
        with open(datadir + fname, "r") as csvfile:
            reader = csv.reader(csvfile)
            i = -1
            total = 0
            print (fname)
            for row in reader:
                if i == -1:
                    i = 0
                    continue
                i += 1
                for j in range(len(row)):
                    if row[j] == "":
                        missing[j] += 1
            totalct += i

    print ("Total count:", totalct)
    print ()

    for i in range(len(HEADER)):
        print ("Missing {} for {}".format(float(missing[i])/totalct, HEADER[i]))


if __name__ == "__main__":
    main()
