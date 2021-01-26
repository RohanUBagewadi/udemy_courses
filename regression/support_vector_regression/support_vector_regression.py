import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

dataset = pd.read_csv('position_salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

# feature scaling
sts_x = StandardScaler()
x = sts_x.fit_transform(x)
sts_y = StandardScaler()  # separate scalar model should be defined and applied to y
y = sts_y.fit_transform(y)

reg = SVR(kernel='rbf')
reg.fit(x, y.reshape(-1))

# reverse scaling for dependent variable
p = sts_y.inverse_transform(reg.predict(sts_x.transform([[6.5]])))

pred = sts_y.inverse_transform(reg.predict(x))

plt.scatter(sts_x.inverse_transform(x), sts_y.inverse_transform(y), color='r')
plt.plot(sts_x.inverse_transform(x), pred, color='b')
plt.title('Simple Vector Regression (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# evaluate the model
r2 = r2_score(sts_y.inverse_transform(y), pred)
print('R^2 score of the model:', r2)

