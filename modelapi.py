import date
import lhapi
import predict


def getDelayForFlight(flNum, origin, date):
    #valid = validateFlight TODO

    model = predict.Model("test", ARR_DELAY, [])
    pred = model.predict(flNum, date, originAirportId)
