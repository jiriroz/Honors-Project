import date
import lhapi
import predict

DBTYPE = "test" #db type will be all
CONN, TABLES, TOTAL = predict.connectToDB(DBTYPE)
MODEL = "temporalLinear15All"

# IATA airport code => DOT airport ID
AIRPORTS = pickle.load(open("keys/airportCodes.p", "rb"))

def getDelayForFlight(flNum, originAirportName, date):
    #valid = validateFlight TODO
    originAirportId = AIRPORTS[originAirportName]

    model = predict.Model(MODEL, ARR_DELAY, [], load=True)
    pred = model.predict(CONN, flNum, date, originAirportId)


delay = getDelayForFlight("QF12", "JFK", date.datetime(2017, 12, 11))

print ("Delay:", delay)

