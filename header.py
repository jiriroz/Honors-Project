import pickle

HEADER = ['YEAR', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER', 'AIRLINE_ID', 'TAIL_NUM', 'FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'DEP_DEL15', 'DEP_DELAY_GROUP', 'DEP_TIME_BLK', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FLIGHTS', 'DISTANCE', 'DISTANCE_GROUP', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

DEP_DELAY = HEADER.index("DEP_DELAY")
ARR_DELAY = HEADER.index("ARR_DELAY")

CRS_ELAPSED_TIME = HEADER.index("CRS_ELAPSED_TIME")
ACTUAL_ELAPSED_TIME = HEADER.index("ACTUAL_ELAPSED_TIME")
DISTANCE = HEADER.index("DISTANCE")

CARRIER_DELAY = HEADER.index("CARRIER_DELAY")
WEATHER_DELAY = HEADER.index("WEATHER_DELAY")
NAS_DELAY = HEADER.index("NAS_DELAY")
SECURITY_DELAY = HEADER.index("SECURITY_DELAY")
LATE_AIRCRAFT_DELAY = HEADER.index("LATE_AIRCRAFT_DELAY")

UNIQUE_CARRIER = HEADER.index("UNIQUE_CARRIER")

YEAR = HEADER.index("YEAR")
MONTH = HEADER.index("MONTH")
DAY_OF_MONTH = HEADER.index("DAY_OF_MONTH")
DAY_OF_WEEK = HEADER.index("DAY_OF_WEEK")

DEP_TIME = HEADER.index("DEP_TIME")
ARR_TIME = HEADER.index("ARR_TIME")
CRS_DEP_TIME = HEADER.index("CRS_DEP_TIME")
CRS_ARR_TIME = HEADER.index("CRS_ARR_TIME")

AIRLINE_ID = HEADER.index("AIRLINE_ID")
ORIGIN_AIRPORT_ID = HEADER.index("ORIGIN_AIRPORT_ID")
DEST_AIRPORT_ID = HEADER.index("DEST_AIRPORT_ID")
ORIGIN_CITY_MARKET_ID = HEADER.index("ORIGIN_CITY_MARKET_ID")
DEST_CITY_MARKET_ID = HEADER.index("DEST_CITY_MARKET_ID")

DELAY_TYPES = [CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY,
               SECURITY_DELAY, LATE_AIRCRAFT_DELAY]

ALL_DELAYS = [DEP_DELAY, ARR_DELAY, CARRIER_DELAY,
              WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY,
              LATE_AIRCRAFT_DELAY]

INT_FEATURES = [YEAR, MONTH, DAY_OF_MONTH, DAY_OF_WEEK,
                CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME, DISTANCE]
FLOAT_FEATURES = []
FLOAT_FEATURES += ALL_DELAYS

"""
Categorical variables:
AIRLINE_ID, ORIGIN_AIRPORT_ID + DEST_AIRPORT_ID,
ORIGIN_CITY_MARKET_ID + DEST_CITY_MARKET_ID
Numerical variables:
CRS_ELAPSED_TIME, DISTANCE
Either (test):
MONTH, DAY_OF_WEEK, CRS_DEP_TIME (hour), CRS_ARR_TIME (hour)
"""
CATEG_VARS = dict()
airline = pickle.load(open("models/airline.p", "rb"))
airport = pickle.load(open("models/airport.p", "rb"))
city = pickle.load(open("models/city.p", "rb"))

CATEG_VARS[AIRLINE_ID] = airline
CATEG_VARS[ORIGIN_AIRPORT_ID] = airport
CATEG_VARS[DEST_AIRPORT_ID] = airport
CATEG_VARS[ORIGIN_CITY_MARKET_ID] = city
CATEG_VARS[DEST_CITY_MARKET_ID] = city

#Dict of index: period
TIME_VARS = dict()
TIME_VARS[MONTH] = 12
TIME_VARS[DAY_OF_WEEK] = 7
TIME_VARS[CRS_DEP_TIME] = 24
TIME_VARS[CRS_ARR_TIME] = 24
