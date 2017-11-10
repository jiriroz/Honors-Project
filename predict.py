import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
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
ALL = "data/all.csv"

def main():
    t = time.time()
    ntrain = 100000
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

    #for win in [10, 12, 14, 16, 18]:
    #    temporal = Model("TemporalModel", ARR_DELAY, [])
    #    temporal.temporalModel(ALL, ntrain, window=win)

    ntrain = 1000000
    temporal = Model("TemporalModel", ARR_DELAY, [])
    temporal.temporalModel(ALL, ntrain, window=15)
    


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

        testRatio = 1.0 / (window)

        flights = dict()
        index = 0

        nData = nExamples * window
        print ("Retrieving", nData)
        for (x, y, row) in iterData(fname, self.features, self.yIndex):
            if index >= nData:
                break
            flNum = getFlNum(row)
            if flNum not in flights:
                flights[flNum] = []
            
            flights[flNum].append((x, y))
            index += 1

        print ("Number of flights: ", len(list(flights.keys())))
        mse = 0

        X_train, Y_train = [], []
        X_test, Y_test = [], []
        X_train_norm, Y_train_norm = [], []
        X_test_norm, Y_test_norm = [], []

        nTest = int(nData/window)
        count = 0

        flNums = list(flights.keys())
        for flNum in flNums:
            if len(flights[flNum]) < window + 1:
                del flights[flNum]

        flNums = list(flights.keys())
        while count < nExamples:
            flNum = random.choice(flNums)
            n = len(flights[flNum])
            index = random.randint(window, n - 1)
            prediction = 0.0
            prev = []
            for j in range(window):
                prev.append(flights[flNum][index - j - 1][1])
                prediction += flights[flNum][index - j - 1][1]
            y = flights[flNum][index][1]

            std = np.std(prev[1:])
            mean = np.mean(prev[1:])
            prevNorm = np.array(prev[:])
            prevNorm -= mean
            yNorm = y - mean
            if std > 0:
                prevNorm /= std
                yNorm /= std
            else:
                continue

            prediction /= window
            #print ("Predicted:", prediction, "actual:", flights[flNum][index][1])
            mse += (prediction - y) ** 2

            if random.random() < 0.8:
                X_train.append(prev)
                Y_train.append(y)
                X_train_norm.append(prevNorm)
                Y_train_norm.append(yNorm)
            else:
                X_test.append(prev)
                Y_test.append(y)
                X_test_norm.append(prevNorm)
                Y_test_norm.append(yNorm)
            count += 1


        """
        mean = 0.0
        std = 0.0
        for prev in X_train:
            mean += np.mean(prev)
            std += np.std(prev)
        mean /= len(X_train)
        std /= len(X_train)
        print ("Mean delay is", mean)
        print ("Mean standard deviation is", std)

        for i in range(10):
            j = random.randint(0, len(X_train) - 1)
            print (Y_train[j], X_train[j])

        means = []
        stds = []
        for prev in X_test:
            means.append(np.mean(prev))
            stds.append(np.std(prev))

        plt.figure("Means")
        plt.hist(means, 50, normed=1)

        plt.figure("Standard deviations")
        plt.hist(stds, 50, normed=1)
        plt.show()
        """
        print ("Window size:", window)

        degree = 2
        poly = PolynomialFeatures(degree=degree)

        X_train = np.array(X_train)
        X_train_poly = poly.fit_transform(X_train)
        Y_train = np.array(Y_train)

        X_test = np.array(X_test)
        X_test_poly = poly.fit_transform(X_test)
        Y_test = np.array(Y_test)

        X_train_norm = np.array(X_train_norm)
        Y_train_norm = np.array(Y_train_norm)
        X_test_norm = np.array(X_test_norm)
        Y_test_norm = np.array(Y_test_norm)

        linear = linear_model.LinearRegression()
        linear.fit(X_train, Y_train)
        pred = linear.predict(X_test)
        mse_linear = 0
        for i in range(len(X_test)):
            mean = np.mean(X_test[i])
            std = np.std(X_test[i])
            Y_test[i] = (Y_test[i] - mean) / std
            pred[i] = (pred[i] - mean) / std
        mse_linear = mean_squared_error(Y_test, pred)

        linear = linear_model.LinearRegression()
        linear.fit(X_train_norm, Y_train_norm)
        pred = linear.predict(X_test_norm)
        mse_linear_norm = mean_squared_error(Y_test_norm, pred)

        #poly = linear_model.LinearRegression()
        #poly.fit(X_train_poly, Y_train)
        #pred = poly.predict(X_test_poly)
        #mse_poly = mean_squared_error(Y_test, pred)
    
        print ("MSE linear", mse_linear)
        print ("MSE linear norm", mse_linear_norm)
        #print ("MSE poly", mse_poly)
        print ()


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
