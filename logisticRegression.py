'''
This file will define the logistic regression cost function,
the gradient, and then minimize via gradient descent.
'''
from scipy import linalg
from scipy.special import expit
import numpy as np


def LRCostFuntion(feature_array, training_array, weights, regularization):
'''
This function defines the cost function of the logistic regression
classifier.
- feature_array is a np.ndarray that belongs to R(nxm), where n is
  the number of featres and m is the number of training examples.
- training_array is a np.ndarray that belongs to R(mx1). This vector
  gives the class (digit) that each set of features represents.
- weights is a np.ndarray that belongs to R(nx1). It is the initial
  weight vector that will be tuned to minimize the cost function.
'''
	# hypothesis will be a mx1 ndarray
	hypothesis = expit((weights.T).dot(feature_array))

	m = feature_array.shape[1]

	J = 1/m * ((training_array.T).dot(np.log(hypothesis)) + 
		(1 - training_array.T).dot(np.log(1 - hypothesis)) +
		regularization/2 * (weights.T).dot(weights))
	# vectorization of the LR cost function:
	# 1/m*sum(y*log(h) + (1-y)*log(1-h)) + lambda/2 theta^2)

	J_gradient = 