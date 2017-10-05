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
CRS_DEP_TIME = HEADER.index("CRS_DEP_TIME")
CRS_ARR_TIME = HEADER.index("CRS_ARR_TIME")

from header import *

DELAYS = [CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY]

REQ_FIELDS = [DEP_TIME, ARR_TIME, CRS_DEP_TIME, CRS_ARR_TIME, DEP_DELAY, ARR_DELAY]

def categoricalVars():
    with open("data/all.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        i = -1
        #AIRLINE_ID, ORIGIN_AIRPORT_ID + DEST_AIRPORT_ID,
        #ORIGIN_CITY_MARKET_ID + DEST_CITY_MARKET_ID
        airline, airlinect = dict(), 0
        airport, airportct = dict(), 0
        city, cityct = dict(), 0
        for row in reader:
            i += 1
            if i == 0:
                continue
            if i % 10000000 == 0:
                print (i)
            if row[AIRLINE_ID] not in airline:
                airline[row[AIRLINE_ID]] = airlinect
                airlinect += 1
            if row[ORIGIN_AIRPORT_ID] not in airport:
                airport[row[ORIGIN_AIRPORT_ID]] = airportct
                airportct += 1
            if row[DEST_AIRPORT_ID] not in airport:
                airport[row[DEST_AIRPORT_ID]] = airportct
                airportct += 1
            if row[ORIGIN_CITY_MARKET_ID] not in city:
                city[row[ORIGIN_CITY_MARKET_ID]] = cityct
                cityct += 1
            if row[DEST_CITY_MARKET_ID] not in city:
                city[row[DEST_CITY_MARKET_ID]] = cityct
                cityct += 1
        print (airlinect)
        print (airportct)
        print (cityct)
        pickle.dump(airline, open("airline.p", "wb"))
        pickle.dump(airport, open("airport.p", "wb"))
        pickle.dump(city, open("city.p", "wb"))
        
                

def aggregate():
    datadir = "data_raw/"
    datafiles = [x for x in os.listdir(datadir) if x[0] != "." and x[-3:] == "csv"]

    totalct = 0
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

    agg_file = open("data/_all.csv", "w")
    agg_writer = csv.writer(agg_file)
    agg_writer.writerow(HEADER[:-1])

    for fname in datafiles:
        #if filect == 1:
        #    break
        filect += 1
        #yr = int(fname[:4])
        #if yr not in years:
        #    continue
        with open(datadir + fname, "r") as csvfile:
            reader = csv.reader(csvfile)
            i = -1
            total = 0.0
            print (fname)
            for row in reader:
                if i == -1:
                    i = 0
                    continue
                if any([row[x] == "" for x in REQ_FIELDS]):
                    continue
                agg_writer.writerow(row[:-1])
                i += 1
            totalct += i
    print ("Wrote {} rows".format(totalct))
    agg_file.close()
    

if __name__ == "__main__":
    categoricalVars()
