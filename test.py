import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Loading the dataset from sklearn toy dataset.
# I am using the argument return_X_y to change the Bunch object to a (data, target) object.
X, y = datasets.load_diabetes(return_X_y=True)

# Because X has many columns, I am using the numpy method "newaxis" to just use 1 feature against the y.
X = X[:, np.newaxis, 2]

# This is the process of splitting the data using the train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=1)

# Training the data with LinearRegression()
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the data on the X_test and assign it to variable
y_pred = regressor.predict(X_test)

# Plotting the data using pyplot
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.show()























