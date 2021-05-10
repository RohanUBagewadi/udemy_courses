import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('50_startups.csv')
x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1:].values

# multiple linear regression avoids dummy variable automatically
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],  remainder='passthrough')
x = ct.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)

pred = lr.predict(x_test)
error = y_test - pred

# evaluate the model
r2 = r2_score(y, lr.predict(x))
print('R^2 score of the model:', r2)



