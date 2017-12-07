import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    exp, poly = [], []
    for i in range(3):
        nTrain = 100000
        nTest = int(nTrain/2)
        model = Model("temporal", ARR_DELAY, [])
        model.temporalModel("train", None, nTrain, degree=2, window=15, train=True, fit=False)
        msePoly, mseExp = model.temporalModel("val", None, nTest, degree=2, window=15, train=False, fit=True)
        poly.append(msePoly)
        exp.append(mseExp)
    print ("Avg mse poly:", np.mean(msePoly))
    print ("Avg mse exp:", np.mean(mseExp))

class Model(object):
    def __init__(self, name, yIndex, features):
        self.yIndex = yIndex
        self.name = name
        self.features = features

    def save(self):
        pickle.dump(self.regr, open("models/{}.p".format(self.name), "wb"))

    def load(self):
        self.regr = pickle.load(open("models/{}.p".format(self.name), "rb"))

    def timeSeriesModel(self, db, table, nExamples, alpha=1, train=False, fit=True):
        fields = ["ARR_DELAY", "PREV_FLIGHT"]
        conn, tables, total = connectToDB(db)
        rows, rowsY = [], []
        tablesPerRow = []
        for row, tblName in dataIterator(table, nExamples, fields, conn, tables, total):
            x, y = preprocess(row, fields, 0)
            rows.append(x)
            rowsY.append(y)
            if table == None:
                tablesPerRow.append(tblName)

        X, Y = [], []
        for i in range(len(rows)):
            tblName = table
            if table == None:
                tblName = tablesPerRow[i]
            prev = getPrevFlights(conn, tblName, rows[i][0], 50, ["ARR_DELAY", "DAY_OF_WEEK"])
            if len(prev) > 0:
                X.append([d[0] for d in prev])
                Y.append(rowsY[i])
            
        X = np.array(X)
        Y = np.array(Y)

        mse = 0
        if fit and len(X) > 0:
            for i in range(len(X)):
                s = X[i][-1]
                for j in range(len(X[i]) - 2, -1, -1):
                    s = alpha * X[i][j] + (1 - alpha) * s
                mse += (s - Y[i]) ** 2
                #print ("X: ", X[i])
                #print ("Y: ", 
                #if i < 3:
                #    plt.plot(X_smooth[i])
            mse /= len(X)

        if train:
            pass
        if fit:
            return mse


    def temporalModel(self, db, table, nExamples, window=1, degree=1, train=False, fit=True):
        fields = ["ARR_DELAY", "PREV_FLIGHT"]
        conn, tables, total = connectToDB(db)
        rows, rowsY = [], []
        tablesPerRow = []
        for row, tblName in dataIterator(table, nExamples, fields, conn, tables, total):
            x, y = preprocess(row, fields, 0)
            rows.append(x)
            rowsY.append(y)
            if table == None:
                tablesPerRow.append(tblName)

        X, Y = [], []
        X_exp, Y_exp = [], []
        for i in range(len(rows)):
            tblName = table
            if table == None:
                tblName = tablesPerRow[i]
            nGet = window
            if fit:
                nGet = -1
            prev = getPrevFlights(conn, tblName, rows[i][0], nGet, ["ARR_DELAY", "DAY_OF_WEEK"])
            if len(prev) >= window:
                X.append([d[0] for d in prev[:window]])
                Y.append(rowsY[i])
            if len(prev) > 2:
                X_exp.append([d[0] for d in prev])
                Y_exp.append(rowsY[i])
            if i < 5 and fit:
                plt.plot([d[0] for d in prev])
                plt.show()

        mse_exp = 0
        if fit and len(X_exp) > 0:
            alpha = 0.2
            for i in range(len(X_exp)):
                s = (X_exp[i][-1] + X_exp[i][-2] + X_exp[i][-3]) / 3
                for j in range(len(X_exp[i]) - 4, -1, -1):
                    s = alpha * X_exp[i][j] + (1 - alpha) * s
                mse_exp += (s - Y_exp[i]) ** 2
            mse_exp /= len(X_exp)

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
            return mse, mse_exp

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
