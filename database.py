import sqlite3
import csv
from header import *
from predict import getFlNum

CREATE = '''CREATE TABLE {}
            (ID INTEGER PRIMARY KEY NOT NULL,
             FL_NUM TEXT NOT NULL,
             AIRLINE_ID INTEGER NOT NULL,
             DEST_AIRPORT_ID INTEGER NOT NULL,
             DISTANCE INTEGER NOT NULL,
             YEAR INTEGER NOT NULL,
             MONTH INTEGER NOT NULL,
             DAY_OF_MONTH INTEGER NOT NULL,
             DAY_OF_WEEK INTEGER NOT NULL,
             CRS_DEP_TIME INTEGER NOT NULL, 
             CRS_ARR_TIME INTEGER NOT NULL,
             DEP_TIME INTEGER NOT NULL,
             ARR_TIME INTEGER NOT NULL,
             CRS_ELAPSED_TIME INTEGER NOT NULL,
             DEP_DELAY REAL NOT NULL,
             ARR_DELAY REAL NOT NULL,
             CARRIER_DELAY REAL,
             WEATHER_DELAY REAL,
             NAS_DELAY REAL, 
             SECURITY_DELAY REAL,
             LATE_AIRCRAFT_DELAY REAL);'''

INSERT = '''INSERT INTO {} (ID, FL_NUM, AIRLINE_ID, DEST_AIRPORT_ID, DISTANCE, YEAR, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, CRS_DEP_TIME, CRS_ARR_TIME, DEP_TIME, ARR_TIME, CRS_ELAPSED_TIME, DEP_DELAY, ARR_DELAY, CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY) 
            VALUES ({}, "{}", {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});'''

tables = dict()

conn = sqlite3.connect("delays.db")

errors = 0

with open("data/all.csv.sorted", "r") as f:
    reader = csv.reader(f)
    first = True
    for row in reader:
        if first:
            first = False
            continue
        tblName = "airport{}".format(row[ORIGIN_AIRPORT_ID])
        if tblName not in tables:
            tables[tblName] = 0
            print ("Create table", tblName)
            conn.execute(CREATE.format(tblName))
        for feat in INT_FEATURES:
            row[feat] = int(float(row[feat]))
        cmd = INSERT.format(tblName, tables[tblName], getFlNum(row), int(row[AIRLINE_ID]),
              int(row[DEST_AIRPORT_ID]), int(row[DISTANCE]), int(row[YEAR]), 
              int(row[MONTH]), int(row[DAY_OF_MONTH]), int(row[DAY_OF_WEEK]),
              int(row[CRS_DEP_TIME]), int(row[CRS_ARR_TIME]), int(row[DEP_TIME]),
              int(row[ARR_TIME]), int(row[CRS_ELAPSED_TIME]), float(row[DEP_DELAY]),
              float(row[ARR_DELAY]), float(row[CARRIER_DELAY]),
              float(row[WEATHER_DELAY]), float(row[NAS_DELAY]),
              float(row[SECURITY_DELAY]), float(row[LATE_AIRCRAFT_DELAY]))
        try:
            conn.execute(cmd)
        except Exception as e:
            print (e)
            print (cmd)
            errors += 1
            continue
        tables[tblName] += 1

conn.commit()
conn.close()
print ("Done")
print ("Errors:", errors)
