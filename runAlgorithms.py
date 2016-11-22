'''
This file imports the train and test data, runs the various 
machine learning algorithms, benchmarks them, and provides 
evaluates their accuracy.
'''

import numpy as np
from visualizeData import visualizeData

# Import train and test data
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

train = np.loadtxt(TRAIN_PATH, dtype = int, delimiter = ',',
	skiprows = 1)
test = np.loadtxt(TEST_PATH, dtype = int, delimiter = ',', 
	skiprows = 1)

# Uncomment to visualize the data as an image
visualizeData(28, 28, train[1][1:])