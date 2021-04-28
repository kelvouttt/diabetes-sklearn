import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

dataset = datasets.load_diabetes()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X['target'] = pd.Series(dataset.target)
y = X['target']

















