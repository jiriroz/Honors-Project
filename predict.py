import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from header import *

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
#TEST_FILE = "data/test.csv" #off limits

def main():
    ntrain = 100000
    ntest = 20000
    feats = [DAY_OF_WEEK] #Can we predict nas delay by day of week?

    model = Model("NasModelLinear", NAS_DELAY, feats)
    model.trainLinearModel(TRAIN_FILE, ntrain)
    model.predictLinearModel(VAL_FILE, ntest)


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
    row[DAY_OF_WEEK] = int(row[DAY_OF_WEEK])
    for feat in FLOAT_FEATURES:
        row[feat] = float(row[feat])
    for feat in INT_FEATURES:
        row[feat] = int(row[feat])
    return row



if __name__ == "__main__":
    main()
