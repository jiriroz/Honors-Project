import numpy as np

class MyOneHotEncoder(object):

    def __init__(self, categVars):
        """
        Initialize the object and set the parameters.
        categVars: List of number of values per categorical feature.
                 If the feature i is not categorical, categVars[i] = 0.
        """
        self.categVars = categVars
        self.rowLength = 0
        for nVal in categVars:
            if nVal == 0:
                self.rowLength += 1
            else:
                self.rowLength += nVal


    def transform(self, row):
        """
        Transform a row of features into a row with the categorical
        features one-hot encoded.
        """
        if len(row) != len(self.categVars):
            raise ValueError("The row contains {} number of values. It is supposed to contain {}".format(len(row), len(self.categVars)))
        transformedRow = np.zeros(self.rowLength)
        index = 0
        for i in range(len(row)):
            if self.categVars[i] == 0:
                transformedRow[index] = row[i]
                index += 1
            else:
                transformedRow[index + row[i]] = 1
                index += self.categVars[i]
        return transformedRow
