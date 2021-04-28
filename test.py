import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

dataset = datasets.load_diabetes(as_frame=True)

X = dataset.data
y = dataset.target

# y = y.to_frame()

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
#
# regressor = linear_model.LinearRegression()
# regressor.fit(X_train, y_train)
#
# plt.scatter(X_train, y_train, color='blue')
# plt.plot(X_train, regressor.predict(X_test), color='red')















