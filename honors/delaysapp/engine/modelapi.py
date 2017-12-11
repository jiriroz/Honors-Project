import datetime
import pickle
import os
import sys

import delaysapp.engine.lhapi as lhapi
import delaysapp.engine.predict as predict
from delaysapp.engine.header import *

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DBTYPE = "test" #db type will be all
modelName = "temporalLinear15All"
#modelName = "temporalPoly15All"
MODEL = predict.Model(modelName, ARR_DELAY, [], load=True)

# IATA airport code => DOT airport ID
AIRPORTS = pickle.load(open(os.path.join(BASE_DIR, "keys/airportCodes.p"), "rb"))

def getDelayForFlight(flNum, originAirportName, date):
    CONN, TABLES, TOTAL = predict.connectToDB(DBTYPE)
    if originAirportName not in AIRPORTS:
        return None, "Airport doesn't exist."
    originAirportId = AIRPORTS[originAirportName]
    pred, error = MODEL.predict(CONN, flNum, date, originAirportId)
    if pred != None:
        error = None
    return pred, error

tests = [("DL472", "JFK"), ("AA33", "JFK"), ("B6123", "JFK"), ("NK224", "ORD"), ("UA1166", "ORD"), ("UA116", "BLA"), ("AA33", "ORD")]

if sys.argv[1] == "runmodule":
    for (flnum, aport) in tests:
        delay, error = getDelayForFlight(flnum, aport, datetime.date(2017, 12, 11))
        if delay != None:
            print ("Delay:", delay)
        else:
            print (error)

