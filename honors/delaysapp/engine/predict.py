import csv
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import pandas as pd
import pickle
import random
from scipy.optimize import minimize
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import sqlite3
import time

from delaysapp.engine.header import *
from delaysapp.engine.one_hot_encoder import MyOneHotEncoder

LOGGING = False

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    """
    #fileAirports = open(os.path.join("data/tablesTrain.p", pFile), "rb")
    #tables = pickle.load(fileAirports)
    print ("Temporal linear")
    nTrain = 3000000
    nTest = int(nTrain/2)
    model = Model("temporalLinear30All", ARR_DELAY, [])
    model.temporalModel("train", None, nTrain, window=30, train=True, fit=False)
    mse = model.temporalModel("val", None, nTest, window=30, train=False, fit=True)
    model.save()
    print (mse)
    """


    """
    n = 10
    model = Model("plotting", "ARR_DELAY", [])
    model.plot("train", None, n)
    model = Model("expSmooth", "ARR_DELAY", [])
    mse = model.timeSeriesModel("train", None, 200000, train=False, fit=True)
    print ("Normal exp smooth")
    print (mse)
    """
    


class Model(object):
    def __init__(self, name, yIndex, features, load=False):
        self.yIndex = yIndex
        self.name = name
        self.features = features
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.idToAirport = pickle.load(open(os.path.join(BASE_DIR, "keys/airportCodesReverse.p"), "rb"))
        if load:
            self.load()

    def save(self):
        name = "temporalPoly15Exp"
        pickle.dump(self.regr2, open(os.path.join(BASE_DIR, "models/{}.p".format(name)), "wb"))
        name = "temporalPoly15ExpDow"
        pickle.dump(self.regr3, open(os.path.join(BASE_DIR, "models/{}.p".format(name)), "wb"))


    def load(self):
        self.regr = pickle.load(open(os.path.join(BASE_DIR, "models/{}.p".format(self.name)), "rb"))

    def plot(self, db, table, nExamples):
        fields = ["ARR_DELAY", "PREV_FLIGHT", "CRS_DEP_TIME"]
        conn, tables, total = connectToDB(db)
        rows, rowsY = [], []
        tablePerRow = []
        for row, tblName in dataIterator(table, nExamples, fields, conn, tables, total):
            x, y = preprocess(row, fields, 0)
            rows.append(x)
            rowsY.append(y)
            if table == None:
                tablePerRow.append(tblName)

        """
        byDow = [[] for i in range(24)]
        for i in range(len(rows)):
            byDow[int(rows[i][1]) - 1].append(rowsY[i])
        #for x in byDow:
        #    print (x)

        avgs = [np.mean(x) for x in byDow]
        print (avgs)
        font = {'family' : 'normal',
                'size'   : 14}

        matplotlib.rc('font', **font)

        barwidth = 0.5
        index = np.arange(24)
        rects1 = plt.bar(index, avgs, barwidth,
                 alpha=0.8,
                 color='#18647D',
                 edgecolor="none")
        plt.title("Average delay per departure hour")
        plt.xticks(index + barwidth / 2, np.arange(24))
        plt.ylabel("Average delay")
        plt.show()
        """

        for i in range(len(rows)):
            tblName = table
            if table == None:
                tblName = tablePerRow[i]
            prev = getPrevFlights(conn, tblName, int(rows[i][0]), -1, ["ARR_DELAY"])
            seq = [d[0] for d in prev]
            s, S = expSmooth(0.2, seq)
            plt.ylabel("Delay in minutes")
            plt.xlabel("Past flights")
            plt.title("Exponential smoothing")
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            plt.plot(seq, color="#006498", linewidth=2)
            plt.plot(S, 'r', linewidth=3)
            plt.show()
            


    def predict(self, conn, flNum, date, originAirportId):
        # Return (delay, origin, dest, ...), error
        if self.name == "temporalLinear30All":
            return self.temporalModelPredict(conn, flNum, date, originAirportId, window=15, degree=1)
        elif self.name == "temporalPoly15All":
            return self.temporalModelPredict(conn, flNum, date, originAirportId, window=15, degree=2)

    def timeSeriesModel(self, db, table, nExamples, alpha=1, gamma=1, train=False, fit=True):
        fields = ["ARR_DELAY", "PREV_FLIGHT", "DAY_OF_WEEK"]
        conn, tables, total = connectToDB(db)
        rows, rowsY = [], []
        tablePerRow = []
        for row, tblName in dataIterator(table, nExamples, fields, conn, tables, total):
            x, y = preprocess(row, fields, 0)
            rows.append(x)
            rowsY.append(y)
            if table == None:
                tablePerRow.append(tblName)

        X, Y = [], []
        X_poly, Y_poly = [], []
        X_polyexp = []
        X_polyexp_dow = []
        DOW = []
        DOW_Y = []
        for i in range(len(rows)):
            tblName = table
            if table == None:
                tblName = tablePerRow[i]
            
            prev = getPrevFlights(conn, tblName, int(rows[i][0]), -1, ["ARR_DELAY", "DAY_OF_WEEK"])
            seq = [d[0] for d in prev]
            days = [int(d[1]) for d in prev]
            if len(prev) > 1:
                X.append([d[0] for d in prev])
                Y.append(rowsY[i])
                DOW.append(days)
                DOW_Y.append(int(rows[i][1]))
            if len(prev) >= 15:
                X_poly.append([d[0] for d in prev[:15]])
                X_polyexp.append([d[0] for d in prev[:15]])
                X_polyexp_dow.append([d[0] for d in prev[:15]])
                Y_poly.append(rowsY[i])
            if len(prev) > 1 and len(prev) < 15:
                X_polyexp.append([])
                X_polyexp_dow.append([])
            #if i < 5 and fit:
                #com = 1 / alpha - 1
                #seqpd = pd.DataFrame(data=seq)
                #ewma = pd.ewma(seqpd, com=com)
                #plt.plot(seq, linewidth=2)
                #plt.plot(pd.ewma(seq, com=com), linewidth=2)
                #plt.plot(days, seq, "o")
                #plt.show()
            
        X = np.array(X)
        Y = np.array(Y)

        mse0 = 0 #Based on dow
        mse1 = 0 #pure exp smoothing
        mse2 = 0 #pure last 10 flights poly
        mse3 = 0 #exp smooting + last 10 flights poly
        mse4 = 0 #exp smooting all days + exp smoothing same dow + last 10 flights poly
        if len(X) > 0:
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                xSmooth = expSmooth(0.2, x)
                mse1 += (xSmooth - y) ** 2
                sameDay = [x[j] for j in range(len(x)) if DOW[i][j] == DOW_Y[i]]
                if sameDay == []:
                    xSmoothDay = xSmooth
                else:
                    xSmoothDay = expSmooth(0.4, sameDay)
                if X_polyexp[i] != []:
                    X_polyexp[i].append(xSmooth)
                if X_polyexp_dow[i] != []:
                    X_polyexp_dow[i].append(xSmooth)
                    X_polyexp_dow[i].append(xSmoothDay)
                mse0 += (xSmoothDay - y) ** 2
            mse1 /= len(X)
            mse0 /= len(X)
            return mse1
        X_polyexp = [x for x in X_polyexp if x != []]
        X_polyexp_dow = [x for x in X_polyexp_dow if x != []]
        X_poly = np.array(X_poly)
        X_polyexp = np.array(X_polyexp)
        X_polyexp_dow = np.array(X_polyexp_dow)
        Y_poly = np.array(Y_poly)

        degree = 2
        if degree > 1:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X_poly)
            X_polyexp = poly.fit_transform(X_polyexp)
            X_polyexp_dow = poly.fit_transform(X_polyexp_dow)
        if train:
            self.regr1 = linear_model.LinearRegression()
            self.regr1.fit(X_poly, Y_poly)

            self.regr2 = linear_model.LinearRegression()
            self.regr2.fit(X_polyexp, Y_poly)

            self.regr3 = linear_model.LinearRegression()
            self.regr3.fit(X_polyexp_dow, Y_poly)
        if fit:
            pred = self.regr1.predict(X_poly)
            mse2 = mean_squared_error(Y_poly, pred)

            pred = self.regr2.predict(X_polyexp)
            mse3 = mean_squared_error(Y_poly, pred)

            pred = self.regr3.predict(X_polyexp_dow)
            mse4 = mean_squared_error(Y_poly, pred)
            return mse1, mse2, mse3, mse4


    def temporalModel(self, db, table, nExamples, window=1, degree=1, train=False, fit=True):
        fields = ["ARR_DELAY", "PREV_FLIGHT"]
        conn, tables, total = connectToDB(db)
        rows, rowsY = [], []
        tablePerRow = []
        for row, tblName in dataIterator(table, nExamples, fields, conn, tables, total):
            x, y = preprocess(row, fields, 0)
            rows.append(x)
            rowsY.append(y)
            if table == None:
                tablePerRow.append(tblName)

        X, Y = [], []
        for i in range(len(rows)):
            tblName = table
            if table == None:
                tblName = tablePerRow[i]
            prev = getPrevFlights(conn, tblName, rows[i][0], window, ["ARR_DELAY", "DAY_OF_WEEK"])
            seq = [d[0] for d in prev]
            if len(prev) >= window:
                X.append(seq[:window])
                Y.append(rowsY[i])

        X = np.array(X)
        Y = np.array(Y)

        if degree > 1:
            poly = PolynomialFeatures(degree=degree)
            X = poly.fit_transform(X)

        if train:
            self.regr = linear_model.LinearRegression()
            self.regr.fit(X, Y)
        if fit:
            pred = self.regr.predict(X)
            r2 = r2_score(Y, pred)
            mse = mean_squared_error(Y, pred)
            return mse

    def temporalModelPredict(self, conn, flNum, date, origin, window=15, degree=1):
        tblName = "airport{}".format(origin)

        query = "SELECT PREV_FLIGHT, DEST_AIRPORT_ID FROM {} WHERE FL_NUM='{}' ORDER BY YEAR DESC, MONTH DESC, DAY_OF_MONTH DESC LIMIT 1"
        cursor = conn.execute(query.format(tblName, flNum))
        res = cursor.fetchone()
        if res == None:
            return None, "Flight doesn't exist at this airport."
        prevRowId = res[0]
        dest = res[1]
        destAirport = self.idToAirport[dest]

        prev = getPrevFlights(conn, tblName, prevRowId, window, ["ARR_DELAY"])
        prev = [d[0] for d in prev]
        if len(prev) < window:
            return None, "Not enough previous flights."
        X = np.array([prev])
        if degree > 2:
            poly = PolynomialFeatures(degree=degree)
            X = poly.fit_transform(X)
        pred = self.regr.predict(X)
        return (pred[0], destAirport), ""

    def linearModel(self, db, table, nExamples, train=False, fit=True):
        conn, tables, total = connectToDB(db)
        X = []
        Y = []
        for row, table in dataIterator(table, nExamples, self.features, conn, tables, total):
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
        for row, table in dataIterator(table, nExamples, self.features, conn, tables, total):
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
        for row, table in dataIterator(table, nExamples, self.features, conn, tables, total):
            mse += row[self.yIndex] ** 2
        mse /= nExamples
        return mse
    
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
            if nTable[table] > tables[table]:
                nTable[table] = tables[table]
            ids = np.random.choice(tables[table], nTable[table], replace=False)
            for rowId in ids:
                cursor = conn.execute(query.format(featNames, table, rowId))
                row = cursor.fetchone()
                result.append((row, table))
        return result
    else:
        repl = n > tables[tableName]
        if n > tables[tableName]:
            n = tables[tableName]
            #print ("Table {} contains only {} rows. Selecting that many.".format(tableName, tables[tableName]))
        ids = np.random.choice(tables[tableName], n, replace=False)
        for rowId in ids:
            cursor = conn.execute(query.format(featNames, tableName, rowId))
            row = cursor.fetchone()
            result.append((row, tableName))
        return result
        
def getPrevFlights(conn, table, prevRowId, window, fields):
    query = "SELECT PREV_FLIGHT, {} FROM {} WHERE ID={}"
    fields = ", ".join(fields)
    result = []
    i = 0
    while (window != -1 and i < window and prevRowId != -1) or (window == -1 and prevRowId != -1):
        cursor = conn.execute(query.format(fields, table, prevRowId))
        row = cursor.fetchone()
        if row == None:
            print ("Reference to prev id doesn't exist: {}, ID {}"
                   .format(table, prevId))
            return result
        prevRowId = row[0]
        result.append(row[1:])
        i += 1
    result = result[::-1]
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
    fileAirports = open(os.path.join(BASE_DIR, pFile), "rb")
    tables = pickle.load(fileAirports)
    fileAirports.close()
    conn = sqlite3.connect(os.path.join(BASE_DIR, dbFile))
    total = sum([tables[x] for x in tables])
    return conn, tables, total

def expSmooth(alpha, x, order=1, gamma=0):
    S = np.empty(len(x), float)
    b = np.empty(len(x), float)
    S[0] = x[0]
    if order > 1:
        b[0] = x[1] - x[0]
    #cutoff = 1.4
    cutoff = 100
    maxVal = 20
    for j in range(1, len(x)):
        if order == 2:
            S[j] = alpha * x[j-1] + (1-alpha) * (S[j-1] + b[j-1])
            b[j] = gamma * (S[j] - S[j-1]) + (1-gamma) * b[j-1]
        else:
            # Don't consider values that are cutoff times the max seen so far
            prevVal = x[j - 1]
            if cutoff:
                if x[j - 1] > cutoff * maxVal:
                    prevVal = maxVal
            S[j] = alpha * prevVal + (1-alpha) * S[j-1]
            maxVal = max(maxVal, prevVal)
    if order == 2:
        return S[-1] + b[-1]
    else:
        return S[-1]

def expSmoothDiff(alpha, x):
    diff = expSmooth(alpha, x) - x
    return np.mean(diff/x)

if sys.argv[1] == "runmodule":
    main()
