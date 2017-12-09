import datetime
import pickle
import lhapi
import predict
from header import *

DBTYPE = "test" #db type will be all
CONN, TABLES, TOTAL = predict.connectToDB(DBTYPE)
MODEL = predict.Model("temporalLinear15All", ARR_DELAY, [], load=True)

# IATA airport code => DOT airport ID
AIRPORTS = pickle.load(open("keys/airportCodes.p", "rb"))

def getDelayForFlight(flNum, originAirportName, date):
    #valid = validateFlight TODO
    if originAirportName not in AIRPORTS:
        return None, "Airport doesn't exist."
    originAirportId = AIRPORTS[originAirportName]
    pred, error = MODEL.predict(CONN, flNum, date, originAirportId)
    return pred, error

tests = [("DL472", "JFK"), ("AA33", "JFK"), ("B6123", "JFK"), ("NK224", "ORD"), ("UA1166", "ORD"), ("UA116", "BLA"), ("AA33", "ORD")]

for (flnum, aport) in tests:
    delay, error = getDelayForFlight(flnum, aport, datetime.date(2017, 12, 11))

    if delay != None:
        print ("Delay:", delay)
    else:
        print (error)

