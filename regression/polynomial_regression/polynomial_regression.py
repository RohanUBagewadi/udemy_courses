import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

dataset = pd.read_csv('position_salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

lr = LinearRegression()
lr.fit(x_poly, y)

pred = lr.predict(x_poly)
error = pred - y

plt.scatter(x, y, color='r')
plt.plot(x, pred, color='b')
plt.title(' Polynomial regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# evaluate the model
r2 = r2_score(y, pred)
print('R^2 score of the model:', r2)


