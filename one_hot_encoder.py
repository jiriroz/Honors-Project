import numpy as np

class MyOneHotEncoder(object):

    def __init__(self, nValues):
        """
        Initialize the object and set the parameters.
        nValues: List of number of values per categorical feature.
                 If the feature i is not categorical, nValues[i] = 0.
        """
        self.nValues = nValues
        self.rowLength = 0
        for nVal in nValues:
            if nVal == 0:
                self.rowLength += 1
            else:
                self.rowLength += nVal


    def transform(self, row):
        """
        Transform a row of features into a row with the categorical
        features one-hot encoded.
        """
        if len(row) != len(self.nValues):
            raise ValueError("The row contains {} number of values. It is supposed to contain {}".format(len(row), len(self.nValues)))
        transformedRow = np.zeros(self.rowLength)
        index = 0
        for i in range(len(row)):
            if self.nValues[i] == 0:
                transformedRow[index] = row[i]
                index += 1
            else:
                transformedRow[index + row[i]] = 1
                index += self.nValues[i]
        return transformedRow
