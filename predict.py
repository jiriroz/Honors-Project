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

TRAIN_DB = "data/train.db"
VAL_DB = "data/val.db"
TEST_DB = "data/test.db"

fileAirports = open("data/tables.p", "rb") #TODO per db
TABLES = pickle.load(fileAirports)
fileAirports.close()
TOTAL_DATA = sum([TABLES[x] for x in TABLES])

def main():

    db = "airport14814"
    print ("Querying ", db)
    conn = sqlite3.connect(TRAIN_DB)
    dataIter = dataIterator(db, 10, "train", ["ID", "FL_NUM", "YEAR", "MONTH"], conn)
    conn.close()
    return


    #ntrain = 1000000
    ntrain = 1000000
    feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, ORIGIN_CITY_MARKET_ID, DEST_CITY_MARKET_ID]

    selected_feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE]

    """
    linear = Model("LinearModel", ARR_DELAY, selected_feats)
    poly = Model("PolynomialModel", ARR_DELAY, selected_feats)

    r2, mse1 = linear.linearModel(TRAIN_FILE, ntrain, train=True, fit=True)
    r2, mse2 = poly.polynomialModel(TRAIN_FILE, ntrain, degree=2, train=True, fit=True)
    r2, mse3 = linear.linearModel(VAL_FILE, ntrain, train=False, fit=True)
    r2, mse4 = poly.polynomialModel(VAL_FILE, ntrain, degree=2, train=False, fit=True)

    print ("Mse linear train:", mse1)
    print ("Mse linear test:", mse3)
    print ("Mse poly train:", mse2)
    print ("Mse poly test:", mse4)
    """

    #temporal = Model("temporal", ARR_DELAY, [])
    #temporal.temporalModel(SORTED_2016, ntrain, window=10)


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

    def temporalModel(self, fname, nExamples, window=1):

        flights = dict()
        index = 0

        nData = int(nExamples * (0.667 * window))
        print ("Retrieving", nData)
        # 10000000 good
        # 15000000 too much
        # 11250000 (0.75 * win size), win=15 perfect
        for (x, y, row) in iterData(fname, self.features, self.yIndex):
            if index >= nData:
                break
            flNum = getFlNum(row)
            if flNum not in flights:
                flights[flNum] = []
            airline = row[AIRLINE_ID]
            airport = row[ORIGIN_AIRPORT_ID]
            flights[flNum].append(((airline, airport), y))
            index += 1

        X_train, Y_train = [], []
        X_test, Y_test = [], []
        nTest = int(nData/window)
        count = 0

        flNums = list(flights.keys())
        for flNum in flNums:
            if len(flights[flNum]) < window + 1:
                del flights[flNum]

        airlines = dict()
        airports = dict()

        flNums = list(flights.keys())
        mse = 0.0
        while count < nExamples:
            flNum = random.choice(flNums)
            n = len(flights[flNum])
            index = random.randint(window, n - 1)
            prev = []
            for j in range(window):
                prev.append(flights[flNum][index - j - 1][1])
            y = flights[flNum][index][1]
            mse += y**2

            aline = flights[flNum][index][0][0]
            aport = flights[flNum][index][0][1]

            if aline not in airlines:
                airlines[aline] = {"train":[[], []], "test":[[], []]}
            if aport not in airports:
                airports[aport] = {"train":[[], []], "test":[[], []]}

            if random.random() < 0.8:
                X_train.append(prev)
                Y_train.append(y)
                airlines[aline]["train"][0].append(prev)
                airlines[aline]["train"][1].append(y)
                airports[aport]["train"][0].append(prev)
                airports[aport]["train"][1].append(y)
            else:
                X_test.append(prev)
                Y_test.append(y)
                airlines[aline]["test"][0].append(prev)
                airlines[aline]["test"][1].append(y)
                airports[aport]["test"][0].append(prev)
                airports[aport]["test"][1].append(y)
            count += 1

        print ("Window size:", window)
        print ("0 predict mse", mse / count)

        degree = 2
        poly = PolynomialFeatures(degree=degree)

        print ("Computing airline models")
        alineMse = []
        for aline in airlines:
            l1 = len(airlines[aline]["train"][0])
            l2 = len(airlines[aline]["test"][0])
            if l2 < 1000:
                continue            
            #print ("Airline", aline)
            #print ("Len train", l1)
            #print ("Len test", l2)
            #print ()
            Xtrain = np.array(airlines[aline]["train"][0])
            Xtrain_poly = poly.fit_transform(Xtrain)
            Ytrain = np.array(airlines[aline]["train"][1])

            Xtest = np.array(airlines[aline]["test"][0])
            Xtest_poly = poly.fit_transform(Xtest)
            Ytest = np.array(airlines[aline]["test"][1])

            regr = linear_model.LinearRegression()
            regr.fit(Xtrain_poly, Ytrain)
            pred = regr.predict(Xtest_poly)
            mse = mean_squared_error(Ytest, pred)
            alineMse.append(mse)
            
        print ("Computing airport models")
        aportMse = []
        for aport in airports:
            l1 = len(airports[aport]["train"][0])
            l2 = len(airports[aport]["test"][0])
            if l2 < 1000:
                continue            
            #print ("Airport", aport)
            #print ("Len train", l1)
            #print ("Len test", l2)
            #print ()
            Xtrain = np.array(airports[aport]["train"][0])
            Xtrain_poly = poly.fit_transform(Xtrain)
            Ytrain = np.array(airports[aport]["train"][1])

            Xtest = np.array(airports[aport]["test"][0])
            Xtest_poly = poly.fit_transform(Xtest)
            Ytest = np.array(airports[aport]["test"][1])

            regr = linear_model.LinearRegression()
            regr.fit(Xtrain_poly, Ytrain)
            pred = regr.predict(Xtest_poly)
            mse = mean_squared_error(Ytest, pred)
            aportMse.append(mse)

        X_train = np.array(X_train)
        X_train_poly = poly.fit_transform(X_train)
        Y_train = np.array(Y_train)

        X_test = np.array(X_test)
        X_test_poly = poly.fit_transform(X_test)
        Y_test = np.array(Y_test)

        print ("Computing linear model")
        regr = linear_model.LinearRegression()
        regr.fit(X_train_poly, Y_train)
        pred = regr.predict(X_test_poly)
        mse_linear = mean_squared_error(Y_test, pred)

        print ("Computing polynomial model")
        regr = linear_model.LinearRegression()
        regr.fit(X_train, Y_train)
        pred = regr.predict(X_test)
        mse_poly = mean_squared_error(Y_test, pred)
    
        #print ("Airline mses", alineMse)
        #print ("Sample airport mses", random.sample(aportMse, 5))
        print ()
        print ("Linear mse", mse_linear)
        print ("Polynomial mse", mse_poly)
        print ("Average airline mse", sum(alineMse) / len(alineMse))
        print ("Average airport mse", sum(aportMse) / len(aportMse))


    def linearModel(self, fname, nExamples, train=False, fit=True):
        X = []
        Y = []
        index = 0
        for (x, y, row) in iterData(fname, self.features, self.yIndex):
            if index >= nExamples:
                break
            X.append(x)
            Y.append(y)
            index += 1
        X = np.array(X)
        Y = np.array(Y)

        if train:
            self.regr = linear_model.LinearRegression()
            self.regr.fit(X, Y)

        if fit:
            pred = self.regr.predict(X)
            r2 = r2_score(Y, pred)
            mse = mean_squared_error(Y, pred)
            return r2, mse

    def polynomialModel(self, fname, nExamples, degree=1, train=False, fit=True):
        X = []
        Y = []
        index = 0
        for (x, y, row) in iterData(fname, self.features, self.yIndex):
            if index >= nExamples:
                break
            X.append(x)
            Y.append(y)
            index += 1
        X = np.array(X)
        Y = np.array(Y)

        poly = PolynomialFeatures(degree=degree)
        X = poly.fit_transform(X)

        if train:
            self.regr = linear_model.LinearRegression()
            self.regr.fit(X, Y)

        if fit:
            pred = self.regr.predict(X)
            r2 = r2_score(Y, pred)
            mse = mean_squared_error(Y, pred)
            return r2, mse

    def dummyModel(self, fname, nExamples):
        index = 0
        mse = 0.0
        for (x, y, row) in iterData(fname, [], self.yIndex):
            if index >= nExamples:
                break
            mse += y ** 2
            index += 1
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

def dataIterator(tableName, n, setType, features, conn): 
    featNames = ", ".join(features)
    criteria = getCriteria(setType)
    query = "SELECT {} FROM {} WHERE ID={} AND {}"
    result = []

    #Get data iterator from the database
    if tableName == None:
        #For now store in array.
        nTable = {tbl:0 for tbl in TABLES}
        tables = list(nTable.keys())
        probs = [TABLES[x]/TOTAL_DATA for x in tables]
        choice = np.random.choice(len(tables), n, p=probs)
        for index in choice:
            nTable[tables[index]] += 1
        for table in TABLES:
            if nTable[table] == 0:
                continue
            # In this unlikely event use replacements
            repl = nTable[table] > TABLES[table]
            ids = np.random.choice(nTable[table], TABLES[table], replace=repl)
            for rowId in ids:
                cursor = conn.execute(query.format(featNames, table, rowId, criteria))
                row = cursor.fetchone()
                result.append(row)
        return result
    else:
        repl = n > TABLES[tableName]
        if n > TABLES[tableName]:
            print ("Table {} contains only {} rows. Selecting with replacement.".format(tableName, TABLES[tableName]))
        ids = np.random.choice(TABLES[tableName], n, replace=repl)
        for rowId in ids:
            print ("Selecting rowID", rowId)
            cursor = conn.execute(query.format(featNames, tableName, rowId, criteria))
            row = cursor.fetchone()
            print (row)
            result.append(row)
        return result
        
def getCriteria(setType):
    if setType == "train":
        return "YEAR < 2016"
    elif setType == "val":
        return "YEAR = 2016"
    elif setType == "test":
        return "YEAR = 2017"
    else:
        raise ValueError("Set has to be train/val/test")

def getPrevFlights(conn, n, rowId):
    pass


def preprocess(row, features, yIndex):
    for feat in FLOAT_FEATURES:
        row[feat] = float(row[feat])
    for feat in INT_FEATURES:
        row[feat] = int(float(row[feat]))
    row[CRS_DEP_TIME] = int(int(row[CRS_DEP_TIME]) / 100) #extract hours
    row[CRS_ARR_TIME] = int(int(row[CRS_ARR_TIME]) / 100)
    for feat in CATEG_VARS:
        #map categorical features so that they are in [0, number of values)
        row[feat] = CATEG_VARS[feat][row[feat]]
    featRow = []
    nValues = []
    for feat in features:
        #print ("Processing feature ", feat)
        if feat in TIME_VARS:
            #print ("Time variable")
            a, b = convertTimeVariable(row[feat], TIME_VARS[feat])
            featRow.append(a)
            featRow.append(b)
            nValues.append(0)
            nValues.append(0)
        elif feat in CATEG_VARS:
            nValues.append(len(CATEG_VARS[feat].keys()))
            featRow.append(row[feat])
            #print ("Categorical variable, n values: ", nValues[-1])
        else:
            #print ("Numerical variable")
            nValues.append(0)
            featRow.append(row[feat])
        
    #print ("Performing one hot encoding")
    #print ("Number of features:", len(nValues))
    #print ("N values", nValues)
    enc = MyOneHotEncoder(nValues=nValues) #make property variable
    featRow = enc.transform(featRow)
    y = row[yIndex]
    return featRow, y, row

def convertTimeVariable(t, period):
    return math.sin(2 * math.pi * t / period), math.cos(2 * math.pi * t / period)

def getFlNum(row):
    return row[UNIQUE_CARRIER].strip() + row[FL_NUM].strip()

if __name__ == "__main__":
    main()
