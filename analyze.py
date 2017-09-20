import csv
import os
import numpy as np
import matplotlib.pyplot as plt

HEADER = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'AIRLINE_ID', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', '']

DEP_DELAY = HEADER.index("DEP_DELAY")
ARR_DELAY = HEADER.index("ARR_DELAY")
CARRIER_DELAY = HEADER.index("CARRIER_DELAY")
WEATHER_DELAY = HEADER.index("WEATHER_DELAY")
NAS_DELAY = HEADER.index("NAS_DELAY")
SECURITY_DELAY = HEADER.index("SECURITY_DELAY")
LATE_AIRCRAFT_DELAY = HEADER.index("LATE_AIRCRAFT_DELAY")

DELAYS = [CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY]

def main():
    datadir = "data/"
    datafiles = [x for x in os.listdir(datadir) if x[0] != "." and x[-3:] == "csv"]

    totalct = 0.0
    poscases = 0
    negcases = 0

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
                if row[ARR_DELAY] == "":
                    continue
                arr_delay = float(row[ARR_DELAY])
                if arr_delay <= 0:
                    continue
                if sum([float(row[x]) for x in DELAYS if row[x] != ""]) == arr_delay:
                    if poscases  % 1000000 == 0:
                        print (arr_delay)
                        print ([float(row[x]) for x in DELAYS if row[x] != ""])
                        print ()
                    poscases += 1
                else:
                    if negcases  % 1000000 == 0:
                        print (arr_delay)
                        print ([float(row[x]) for x in DELAYS if row[x] != ""])
                        print ()
                    negcases += 1
            totalct += i

    print (poscases, negcases)



if __name__ == "__main__":
    main()
