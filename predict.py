import csv
import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

from header import *

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
#TEST_FILE = "data/test.csv" #off limits
SMALL_FILE = "sample.csv"
ONE_ROW = "onerow.csv"

def main():
    ntrain = 100000
    ntest = 20000
    feats = [DAY_OF_WEEK] #Can we predict nas delay by day of week?

    model = Model("NasModelLinear", NAS_DELAY, feats)
    model.trainLinearModel(ONE_ROW, ntrain)
    #model.predictLinearModel(VAL_FILE, ntest)


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

        X = np.zeros((nExamples, len(self.features)), dtype=np.float64)
        Y = np.zeros(nExamples, dtype=np.float64)
        n = 0
        for row in iterData(fname):
            if n >= nExamples:
                break
            print (row)
            Y[n] = row[self.yIndex]
            for i in range(len(self.features)):
                X[n][i] = row[self.features[i]]
            n += 1
        self.regr = linear_model.LinearRegression()
        self.regr.fit(X, Y)

    def predictLinearModel(self, fname, nExamples):
        X = np.zeros((nExamples, len(self.features)), dtype=np.float64)
        Y = np.zeros(nExamples, dtype=np.float64)
        n = 0
        for row in iterData(fname):
            if n >= nExamples:
                break
            Y[n] = row[self.yIndex]
            for i in range(len(self.features)):
                X[n][i] = row[self.features[i]]
            n += 1
        pred = self.regr.predict(X)
        print("Mean squared error: %.2f"
              % mean_squared_error(Y, pred))
        # 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(Y, pred))

def iterData(fname):
    #Helper generator to read from csv files
    #Return preprocessed row
    with open(fname, "r") as csvfile:
        reader = csv.reader(csvfile)
        first = True
        for row in reader:
            if first:
                first = False
                continue
            yield preprocess(row)

def preprocess(row):
    for feat in FLOAT_FEATURES:
        row[feat] = float(row[feat])
    for feat in INT_FEATURES:
        row[feat] = int(float(row[feat]))
    #convert categorical variables to indicator variables
    for feat in CATEG_VARS:
        row[feat] = CATEG_VARS[feat][row[feat]]
    row[CRS_DEP_TIME] = int(row[CRS_DEP_TIME]) / 100 #extract hours
    row[CRS_ARR_TIME] = int(row[CRS_ARR_TIME]) / 100
    featRow = []
    categIndices = []
    nValues = []
    for feat in ALL_FEATURES:
        print ("Processing feature ", feat)
        if feat in TIME_VARS:
            print ("Time variable")
            a, b = convertTimeVariable(row[feat], TIME_VARS[feat])
            featRow.append(a)
            featRow.append(b)
            nValues.append(0)
            nValues.append(0)
        elif feat in CATEG_VARS:
            categIndices.append(len(row) - 1)
            nValues.append(len(CATEG_VARS[feat].keys()))
            featRow.append(row[feat])
            print ("Categorical variable, n values: ", nValues[-1])
        else:
            print ("Numerical variable")
            nValues.append(0)
            featRow.append(row[feat])
        
    enc = OneHotEncoder(n_values=nValues, categorical_features=categIndices) #make property variable
    featRow = enc.transform([featRow])[0]
        
    return featRow

def convertTimeVariable(t, period):
    return math.sin(2 * math.pi * t / period), math.cos(2 * math.pi * t / period)


if __name__ == "__main__":
    main()
