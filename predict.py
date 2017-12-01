import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import sqlite3
import time

from header import *
from one_hot_encoder import MyOneHotEncoder

LOGGING = False

def main():

    table = "airport14679"
    #table = None
    print ("Table", table)
    
    n = 10
    #model = Model("model", 0, ["ARR_DELAY", "AIRLINE_ID", "DISTANCE"])
    #mse = model.linearModel("train", table, n, train=True, fit=True)
    #mse = model.polynomialModel("train", table, n, train=True, fit=True)

    model = Model("temporal", ARR_DELAY, [])
    mse = model.temporalModel("train", table, n, window=10, train=True, fit=True)
    print (mse)

    return

    selected_feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE]



class Predictor(object):

    def __init__(self):
        pass

    def evaluate(self, fname):
        """
        @param fname csv filename containing flight data
        Evaluate our prediction on this dataset.
        @return an error measure (TBD).
        """
        pass

    def predict(self, fname):
        """
        @param fname csv filename containing flight data
        Compute expected delay for each flight.
        """
        delays = []
        with open(fname, "r") as csvfile:
            reader = csv.reader(csvfile)
            i = -1
            for row in reader:
                if i == -1:
                    i = 0
                    continue
                i += 1
                delay = compDelayForFlight(row)
                delays.append(delay)
        delays = np.array(delay)
        return delays

    def compDelayForFlight(self, flightData):
        carrier = self.compCarrierDelay(flightData)
        weather = self.compExtrWeatherDelay(flightData)
        nas = self.compNasDelay(flightData)
        security = self.compSecurityDelay(flightData)
        late = self.compLateAircraftDelay(flightData)
        return carrier + weather + nas + security + late
       
    def compCarrierDelay(self, flightData):
        return 0.0

    def compExtrWeatherDelay(self, flightData):
        return 0.0

    def compNasDelay(self, flightData):
        return 0.0

    def compSecurityDelay(self, flightData):
        return 0.0

    def compLateAircraftDelay(self, flightData):
        return 0.0

class Model(object):
    def __init__(self, name, yIndex, features):
        self.yIndex = yIndex
        self.name = name
        self.features = features

    def save(self):
        pickle.dump(self.regr, open("models/{}.p".format(self.name), "wb"))

    def load(self):
        self.regr = pickle.load(open("models/{}.p".format(self.name), "rb"))

    def temporalModel(self, db, table, nExamples, window=1, train=False, fit=True):
        fields = ["ARR_DELAY", "PREV_FLIGHT"]
        conn, tables, total = connectToDB(db)
        rows, rowsY = [], []
        for row in dataIterator(table, nExamples, fields, conn, tables, total):
            x, y = preprocess(row, fields, 0)
            rows.append(x)
            rowsY.append(y)

        X, Y = [], []
        for i in range(len(rows)):
            #TODO what if table is null?
            prev = getPrevFlights(conn, table, rows[i][0], window, ["ARR_DELAY"])
            if prev != None:
                X.append([d[0] for d in prev])
                Y.append(rowsY[i])
            
        for i in range(len(X)):
            print (Y[i], X[i])
            
        X = np.array(X)
        Y = np.array(Y)
        print ("Valid data points:", len(Y))
        print()

        #poly = PolynomialFeatures(degree=2)
        #X = poly.fit_transform(X)

        if train:
            self.regr = linear_model.LinearRegression()
            self.regr.fit(X, Y)
        if fit:
            pred = self.regr.predict(X)
            r2 = r2_score(Y, pred)
            mse = mean_squared_error(Y, pred)
        return mse

    def linearModel(self, db, table, nExamples, train=False, fit=True):
        conn, tables, total = connectToDB(db)
        X = []
        Y = []
        for row in dataIterator(table, nExamples, self.features, conn, tables, total):
            x, y = preprocess(row, self.features, self.yIndex)
            X.append(x)
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)

        if train:
            self.regr = linear_model.LinearRegression()
            self.regr.fit(X, Y)

        if fit:
            pred = self.regr.predict(X)
            r2 = r2_score(Y, pred)
            mse = mean_squared_error(Y, pred)
            return mse

    def polynomialModel(self, db, table, nExamples, train=False, fit=True):
        conn, tables, total = connectToDB(db)
        X = []
        Y = []
        for row in dataIterator(table, nExamples, self.features, conn, tables, total):
            x, y = preprocess(row, self.features, self.yIndex)
            X.append(x)
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)

        poly = PolynomialFeatures(degree=2)
        X = poly.fit_transform(X)

        if train:
            self.regr = linear_model.LinearRegression()
            self.regr.fit(X, Y)

        if fit:
            pred = self.regr.predict(X)
            r2 = r2_score(Y, pred)
            mse = mean_squared_error(Y, pred)
            return mse

    def dummyModel(self, db, table, nExamples):
        conn, tables, total = connectToDB(db)
        mse = 0.0
        for row in dataIterator(table, nExamples, self.features, conn, tables, total):
            mse += row[self.yIndex] ** 2
        mse /= nExamples
        return mse
    

def iterDataCsv(fname, features, yIndex):
    #Helper generator to read from csv files
    #Return preprocessed row as well as the original row
    with open(fname, "r") as csvfile:
        reader = csv.reader(csvfile)
        first = True
        for row in reader:
            if first:
                first = False
                continue
            yield preprocess(row, features, yIndex)

def dataIterator(tableName, n, features, conn, tables, totalData): 
    featNames = ", ".join(features)
    query = "SELECT {} FROM {} WHERE ID={}"
    result = []

    #Get data iterator from the database
    if tableName == None:
        #For now store in array.
        nTable = {tbl:0 for tbl in tables}
        tableNames = list(nTable.keys())
        probs = [tables[x]/totalData for x in tableNames]
        choice = np.random.choice(len(tableNames), n, p=probs)
        for index in choice:
            nTable[tableNames[index]] += 1
        for table in tables:
            if nTable[table] == 0:
                continue
            # In this unlikely event use replacements
            repl = nTable[table] > tables[table]
            ids = np.random.choice(tables[table], nTable[table], replace=repl)
            for rowId in ids:
                cursor = conn.execute(query.format(featNames, table, rowId))
                row = cursor.fetchone()
                result.append(row)
        return result
    else:
        repl = n > tables[tableName]
        if n > tables[tableName]:
            print ("Table {} contains only {} rows. Selecting with replacement.".format(tableName, tables[tableName]))
        ids = np.random.choice(tables[tableName], n, replace=repl)
        for rowId in ids:
            cursor = conn.execute(query.format(featNames, tableName, rowId))
            row = cursor.fetchone()
            result.append(row)
        return result
        
def getPrevFlights(conn, table, prevRowId, window, fields):
    query = "SELECT PREV_FLIGHT, {} FROM {} WHERE ID={}"
    fields = ", ".join(fields)
    result = []
    for i in range(window):
        if prevRowId == -1:
            return None
        cursor = conn.execute(query.format(fields, table, prevRowId))
        row = cursor.fetchone()
        if row == None:
            print ("Reference to prev id doesn't exist: {}, ID {}"
                   .format(table, prevId))
            return None
        prevRowId = row[0]
        result.append(row[1:])
    return result

def preprocess(row, header, yIndex):
    featRow = []
    categVars = []
    for i in range(len(header)):
        if i == yIndex:
            continue
        value = row[i] # Row is a tuple which cannot be changed
        if header[i] == "CRS_DEP_TIME" or header[i] == "CRS_ARR_TIME":
            value = int(int(value) / 100)
        if header[i] in TIME_VARS:
            #print ("Time variable")
            a, b = convertTimeVariable(value, TIME_VARS[header[i]])
            featRow.append(a)
            featRow.append(b)
            categVars.append(0)
            categVars.append(0)
        elif header[i] in CATEG_VARS:
            #map categorical features so that they are in [0, number of values)
            value = CATEG_VARS[header[i]][value]
            categVars.append(len(CATEG_VARS[header[i]].keys()))
            featRow.append(value)
            #print ("Categorical variable, n values: ", categVars[-1])
        else:
            #print ("Numerical variable")
            categVars.append(0)
            featRow.append(value)
            
    enc = MyOneHotEncoder(categVars=categVars)
    featRow = enc.transform(featRow)
    y = row[yIndex]
    return featRow, y

def convertTimeVariable(t, period):
    return math.sin(2 * math.pi * t / period), math.cos(2 * math.pi * t / period)

def getFlNum(row):
    return row[UNIQUE_CARRIER].strip() + row[FL_NUM].strip()

def connectToDB(dbType):
    trainDB, trainTables = "data/train.db", "data/tablesTrain.p"
    valDB, valTables = "data/val.db", "data/tablesVal.p"
    testDB, testTables = "data/test.db", "data/tablesTest.p"

    if dbType == "train":
        dbFile = trainDB
        pFile = trainTables
    elif dbType == "val":
        dbFile = valDB
        pFile = valTables
    elif dbType == "test":
        dbFile = testDB
        pFile = testTables
    else:
        print ("Unknown database type", dbType)
        return None, None
    fileAirports = open(pFile, "rb")
    tables = pickle.load(fileAirports)
    fileAirports.close()
    conn = sqlite3.connect(dbFile)
    total = sum([tables[x] for x in tables])
    return conn, tables, total

if __name__ == "__main__":
    main()
