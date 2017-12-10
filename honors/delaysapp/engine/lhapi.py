import http.client
import json

TOKEN = "zrfqp7thjhxefm98xgnpne6q"

def getSchedules(origin, dest, datetime):
    params = ""
    url = "/v1/operations/schedules/{}/{}/{}?directFlights=1"
    url = url.format(origin, dest, datetime)

    headers = {"Content-type": "application/json",
               "Accept": "application/json",
               "Authorization":"Bearer {}".format(TOKEN)} 
    conn = http.client.HTTPSConnection("api.lufthansa.com", 443)
    conn.request("GET", url, params, headers)
    resp = conn.getresponse()
    content = resp.read().decode("utf-8")
    parsed = json.loads(content)

    return parsed["ScheduleResource"]["Schedule"]

def getOriginAirport(flNum, date):
    params = ""
    url = "/v1/operations/flightstatus/{}/{}"
    url = url.format(flNum, date.isoformat())

    headers = {"Content-type": "application/json",
               "Accept": "application/json",
               "Authorization":"Bearer {}".format(TOKEN)} 
    conn = http.client.HTTPSConnection("api.lufthansa.com", 443)
    conn.request("GET", url, params, headers)
    resp = conn.getresponse()
    content = resp.read().decode("utf-8")
    parsed = json.loads(content)
    if "FlightStatusResource" in parsed:
        flightStatus = parsed["FlightStatusResource"]["Flights"]["Flight"][0]
        htStatus = parsed["FlightStatusResource"]["Flights"]["Flight"][0]
    else:
        return None

if __name__ == "__main__":
    '''origin = "JFK"
    dest = "LAX"
    datetime = "2017-12-09"
    schedules = getSchedules(origin, dest, datetime)
    for sch in schedules:
        print (sch)'''
    import datetime
    res = getOriginAirport("QF12", datetime.date(2017, 12, 9))
    print (res)
