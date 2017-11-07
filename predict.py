import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import time

from header import *
from one_hot_encoder import MyOneHotEncoder

LOGGING = False

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
#TEST_FILE = "data/test.csv" #off limits
SMALL_FILE = "sample.csv"
ONE_ROW = "onerow.csv"
SORTED = "data/2016.csv"

def main():
    t = time.time()
    ntrain = 500000
    ntest = 100
    feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, ORIGIN_CITY_MARKET_ID, DEST_CITY_MARKET_ID]

    selected_feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE]

    model = Model("TemporalModel", ARR_DELAY, [DISTANCE])

    model.temporalModel(SORTED, ntrain, window=12)


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

        testRatio = 1.0 / (window + 3)

        flights = dict()
        index = 0

        nData = int((1 / testRatio) * nExamples)
        for (x, y, row) in iterData(fname, self.features, self.yIndex):
            if index >= nData:
                break
            flNum = getFlNum(row)
            if flNum not in flights:
                flights[flNum] = []
            
            flights[flNum].append((x, y))
            index += 1

        print ("Number of flights: ", len(list(flights.keys())))
        mse1 = 0
        mse2 = 0
        count = 0

        X_train, Y_train = [], []
        X_test, Y_test = [], []
        
        for flNum in flights:
            n = len(flights[flNum])
            if n < window + 1:
                continue
            nTest = int(n * testRatio)
            for i in range(nTest):
                count += 1
                index = random.randint(window, n - 1)
                prediction1 = 0.0
                prediction2 = 0.0
                prev = []
                for j in range(window):
                    prev.append(flights[flNum][index - j - 1][1])
                    w = 0.8
                    if j < int(window / 2):
                        w = 1.2
                    prediction1 += flights[flNum][index - j - 1][1]
                    prediction2 += flights[flNum][index - j - 1][1] * w
                prediction1 /= window
                prediction2 /= window
                #print ("Predicted:", prediction, "actual:", flights[flNum][index][1])
                mse1 += (prediction1 - flights[flNum][index][1]) ** 2
                mse2 += (prediction2 - flights[flNum][index][1]) ** 2

                if random.random() < 0.8:
                    X_train.append(prev)
                    Y_train.append(flights[flNum][index][1])
                else:
                    X_test.append(prev)
                    Y_test.append(flights[flNum][index][1])


        mse1 /= count
        mse2 /= count
        print ("Count", count)
        print ("Window size:", window)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        regr = linear_model.LinearRegression()
        regr.fit(X_train, Y_train)
        pred = regr.predict(X_train)
        r2_train = r2_score(Y_train, pred)
        mse_train = mean_squared_error(Y_train, pred)

        pred = regr.predict(X_test)
        r2_test = r2_score(Y_test, pred)
        mse_test = mean_squared_error(Y_test, pred)
    
        print ("MSE train", mse_train)
        print ("R2 train", r2_train)
        print ("MSE test", mse_test)
        print ("R2 test", r2_test)


        print ()

    def trainLinearModel(self, fname, nExamples):
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

        self.categ = dict()
        for i in range(len(X)):
            row = X[i]
            y = Y[i]
            j = 0
            for j in range(len(row)):
                if row[j] == 1:
                    break
            if j not in self.categ:
                self.categ[j] = [0, 0.0]
            self.categ[j][0] += 1
            if y > 15:
                self.categ[j][1] += 1
        #for key in self.categ:
        #    self.categ[key][1] /= self.categ[key][0]
        #    print (key, "Probability of delay", self.categ[key][1])

        if LOGGING:
            print ("X", X)
            print ("Y", Y)

        regr = linear_model.LinearRegression()
        regr.fit(X, Y)

    def predictLinearModel(self, fname, nExamples, retResult=False):
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

        categ = dict()
        for i in range(len(X)):
            row = X[i]
            y = Y[i]
            j = 0
            for j in range(len(row)):
                if row[j] == 1:
                    break
            if j not in categ:
                categ[j] = [0, 0]
            categ[j][0] += 1
            if y > 15:
                categ[j][1] += 1
        #for key in categ:
        #    categ[key][1] /= categ[key][0]
        #    print (key, "Probability of delay", categ[key][1])

        pred = self.regr.predict(X)
        r2 = r2_score(Y, pred)
        mse = mean_squared_error(Y, pred)
        if retResult:
            return r2, mse, pred
        else:
            return r2, mse

def iterData(fname, features, yIndex):
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
    return row[4].strip() + row[7].strip()

if __name__ == "__main__":
    main()
