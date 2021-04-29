import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Loading the dataset from sklearn toy dataset.
# I am using the argument return_X_y to change the Bunch object to a (
X, y = datasets.load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)





















