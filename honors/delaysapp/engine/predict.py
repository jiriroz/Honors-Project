import csv
import math
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

    nTrain = 300
    nTest = int(nTrain/2)
    model = Model("temporalPoly15All", ARR_DELAY, [])
    model.temporalModel("train", None, nTrain, window=15, train=True, fit=False)
    mse = model.temporalModel("val", None, nTest, window=15, train=False, fit=True)
    print (mse)

class Model(object):
    def __init__(self, name, yIndex, features, load=False):
        self.yIndex = yIndex
        self.name = name
        self.features = features
        if load:
            self.load()

    def save(self):
        pickle.dump(self.regr, open(os.path.join(BASE_DIR, "models/{}.p".format(self.name)), "wb"))

    def load(self):
        self.regr = pickle.load(open(os.path.join(BASE_DIR, "models/{}.p".format(self.name)), "rb"))

    def predict(self, conn, flNum, date, originAirportId):
        if self.name == "temporalLinear15All":
            return self.temporalModelPredict(conn, flNum, date, originAirportId, window=15, degree=1)
        elif self.name == "temporalPoly15All":
            return self.temporalModelPredict(conn, flNum, date, originAirportId, window=15, degree=2)

    def timeSeriesModel(self, db, table, nExamples, alpha=1, gamma=1, train=False, fit=True):
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
            prev = getPrevFlights(conn, tblName, rows[i][0], -1, ["ARR_DELAY", "DAY_OF_WEEK"])
            seq = [d[0] for d in prev]
            days = [d[1] for d in prev]
            if len(prev) > 1:
                X.append([d[0] for d in prev])
                Y.append(rowsY[i])
            if i < 5 and fit:
                com = 1 / alpha - 1
                seqpd = pd.DataFrame(data=seq)
                ewma = pd.ewma(seqpd, com=com)
                #plt.plot(seq, linewidth=2)
                #plt.plot(pd.ewma(seq, com=com), linewidth=2)
                #plt.plot(days, seq, "o")
                #plt.show()
            
        X = np.array(X)
        Y = np.array(Y)

        mse1 = 0
        mse2 = 0
        if fit and len(X) > 0:
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                xSmooth1 = expSmooth(alpha, x)
                xSmooth2 = expSmooth(alpha, x, order=2, gamma=gamma)
                mse1 += (xSmooth1 - y) ** 2
                mse2 += (xSmooth2 - y) ** 2
                #plt.plot(xSmooth, linewidth=1.5)
                #plt.show()
            mse1 /= len(X)
            mse2 /= len(X)

        if train:
            pass
        if fit:
            return mse1, mse2


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

        query = "SELECT PREV_FLIGHT FROM {} WHERE FL_NUM='{}' ORDER BY YEAR DESC, MONTH DESC, DAY_OF_MONTH DESC LIMIT 1"
        cursor = conn.execute(query.format(tblName, flNum))
        res = cursor.fetchone()
        if res == None:
            return None, "Flight doesn't exist at this airport."
        prevRowId = res[0]

        prev = getPrevFlights(conn, tblName, prevRowId, window, ["ARR_DELAY"])
        prev = [d[0] for d in prev]
        if len(prev) < window:
            return None, "Not enough previous flights."
        X = np.array([prev])
        if degree > 2:
            poly = PolynomialFeatures(degree=degree)
            X = poly.fit_transform(X)
        pred = self.regr.predict(X)
        return pred[0], ""

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
    b[0] = x[1] - x[0]
    for j in range(1, len(x)):
        if order == 2:
            S[j] = alpha * x[j-1] + (1-alpha) * (S[j-1] + b[j-1])
            b[j] = gamma * (S[j] - S[j-1]) + (1-gamma) * b[j-1]
        else:
            S[j] = alpha * x[j-1] + (1-alpha) * S[j-1]
    if order == 2:
        return S[-1] + b[-1]
    else:
        return S[-1]

def expSmoothDiff(alpha, x):
    diff = expSmooth(alpha, x) - x
    return np.mean(diff/x)

if sys.argv[1] == "runmodule":
    main()
