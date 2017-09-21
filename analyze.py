import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

HEADER = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'AIRLINE_ID', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', '']

DEP_DELAY = HEADER.index("DEP_DELAY")
ARR_DELAY = HEADER.index("ARR_DELAY")
CARRIER_DELAY = HEADER.index("CARRIER_DELAY")
WEATHER_DELAY = HEADER.index("WEATHER_DELAY")
NAS_DELAY = HEADER.index("NAS_DELAY")
SECURITY_DELAY = HEADER.index("SECURITY_DELAY")
LATE_AIRCRAFT_DELAY = HEADER.index("LATE_AIRCRAFT_DELAY")
UNIQUE_CARRIER = HEADER.index("UNIQUE_CARRIER")
MONTH = HEADER.index("MONTH")
DAY_OF_WEEK = HEADER.index("DAY_OF_WEEK")
DEP_TIME = HEADER.index("DEP_TIME")
ARR_TIME = HEADER.index("ARR_TIME")

DELAYS = [CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY]

def main():
    datadir = "data/"
    datafiles = [x for x in os.listdir(datadir) if x[0] != "." and x[-3:] == "csv"]

    totalct = 0.0
    poscases = 0
    negcases = 0

    years = {2012:0.0, 2013: 0.0, 2014: 0.0, 2015: 0.0, 2016: 0.0}
    yearsct = {2012:0.0, 2013: 0.0, 2014: 0.0, 2015: 0.0, 2016: 0.0}

    carrier_delay = []
    weather_delay = []
    nas_delay = []
    security_delay = []
    late_aircraft_delay = []

    filect = 0
    for fname in datafiles:
        #if filect == 1:
        #    break
        filect += 1
        yr = int(fname[:4])
        if yr not in years:
            continue
        with open(datadir + fname, "r") as csvfile:
            reader = csv.reader(csvfile)
            i = -1
            total = 0.0
            totalct = 0
            print (fname)
            for row in reader:
                if i == -1:
                    i = 0
                    continue
                i += 1
                if row[DEP_TIME] == "" or row[ARR_TIME] == "":
                    continue
                if all([row[x] == "" for x in DELAYS]):
                    continue
                delays = dict()
                for x in DELAYS:
                    if row[x] == "":
                        delays[x] = 0.0
                    else:
                        delays[x] = float(row[x])
                carrier_delay.append(delays[CARRIER_DELAY])
                weather_delay.append(delays[WEATHER_DELAY])
                nas_delay.append(delays[NAS_DELAY])
                security_delay.append(delays[SECURITY_DELAY])
                late_aircraft_delay.append(delays[LATE_AIRCRAFT_DELAY])
                
                arr_delay = 0.0
                if row[ARR_DELAY] != "":
                    arr_delay = float(row[ARR_DELAY])
                dep_delay = 0.0
                if row[DEP_DELAY] != "":
                    dep_delay = float(row[DEP_DELAY])
                total += arr_delay
                
            totalct += i

    carrier_delay = np.array(carrier_delay)
    weather_delay = np.array(weather_delay)
    nas_delay = np.array(nas_delay)
    security_delay = np.array(security_delay)
    late_aircraft_delay = np.array(late_aircraft_delay)

    delays = [
    carrier_delay,
    weather_delay, 
    nas_delay, 
    security_delay, 
    late_aircraft_delay]
    delays = np.array(delays)
    
    del_names = ["carrier", "weather", "nas", "security", "late"]
    for i in range(5):
        for j in range(i):
            corr = np.corrcoef(delays[i], delays[j])[0][1]
            print ("{} with {}: {}".format(del_names[i], del_names[j], corr))


if __name__ == "__main__":
    main()
