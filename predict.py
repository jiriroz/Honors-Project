import csv
import math
import matplotlib.pyplot as plt
import numpy as np
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

def main():
    ntrain = 100000
    ntest = 100000
    feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, DISTANCE, AIRLINE_ID, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, ORIGIN_CITY_MARKET_ID, DEST_CITY_MARKET_ID]
    for delay in DELAY_TYPES:
        for feat in feats:
            t = time.time()
            model = Model("ModelLinear", delay, [feat])
            model.trainLinearModel(TRAIN_FILE, ntrain)
            r2, mse = model.predictLinearModel(VAL_FILE, ntest)
            print ("Feature {} on {}".format(featName(feat), featName(delay)))
            print ("MSE: {}".format(mse))
        print ()

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

    def trainLinearModel(self, fname, nExamples):
        X = []
        Y = []
        index = 0
        for (x, y) in iterData(fname, self.features, self.yIndex):
            if index >= nExamples:
                break
            X.append(x)
            Y.append(y)
            index += 1
        X = np.array(X)
        Y = np.array(Y)

        self.categ = dict()
        visualize = []
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
            self.categ[j][1] += y
            if j == 18:
                visualize.append(y)
        for key in self.categ:
            self.categ[key][1] /= self.categ[key][0]
            if LOGGING:
                print (key, self.categ[key][1])

        if LOGGING:
            print ("X", X)
            print ("Y", Y)
        self.regr = linear_model.LinearRegression()
        self.regr.fit(X, Y)

    def predictLinearModel(self, fname, nExamples):
        X = []
        Y = []
        index = 0
        for (x, y) in iterData(fname, self.features, self.yIndex):
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
            categ[j][1] += y
        for key in categ:
            categ[key][1] /= categ[key][0]
            if LOGGING:
                print (key, categ[key][1])

        pred = self.regr.predict(X)
        r2 = r2_score(Y, pred)
        mse = mean_squared_error(Y, pred)
        return r2, mse

def iterData(fname, features, yIndex):
    #Helper generator to read from csv files
    #Return preprocessed row
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
    return featRow, y

def convertTimeVariable(t, period):
    return math.sin(2 * math.pi * t / period), math.cos(2 * math.pi * t / period)


if __name__ == "__main__":
    main()
