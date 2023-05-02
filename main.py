import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# separate data into X and Y
data = pd.read_csv('mushrooms.csv')
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# separate data into training and test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
