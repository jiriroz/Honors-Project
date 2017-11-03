import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
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
    t = time.time()
    ntrain = 5000000
    ntest = 1000000
    feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE, ORIGIN_AIRPORT_ID, DEST_AIRPORT_ID, ORIGIN_CITY_MARKET_ID, DEST_CITY_MARKET_ID]

    selected_feats = [MONTH, DAY_OF_WEEK, CRS_ELAPSED_TIME, CRS_DEP_TIME, CRS_ARR_TIME, AIRLINE_ID, DISTANCE]

    total_delay = np.zeros(ntest)

    for DELAY in DELAY_TYPES:
        print ("N train {}, N test {}".format(ntrain, ntest))
        model = Model("SimpleLinearModel", ARR_DELAY, selected_feats)
        model.trainLinearModel(TRAIN_FILE, ntrain)
        r2, mse, pred = model.predictLinearModel(VAL_FILE, ntest, retResult=True)
        total_delay += pred
    
    Y = []
    index = 0
    for (x, y) in iterData(VAL_FILE, [0], ARR_DELAY):
        if index >= ntest:
            break
        Y.append(y)
        index += 1
    Y = np.array(Y)

    mse = sum((Y - total_delay) ** 2) / ntest
    print ("MSE:", mse)


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

        self.regr = linear_model.LinearRegression()
        self.regr.fit(X, Y)

    def predictLinearModel(self, fname, nExamples, retResult=False):
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
