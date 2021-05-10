import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

dataset = pd.read_csv('position_salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

reg = DecisionTreeRegressor(random_state=0)
reg.fit(x, y)

pred = reg.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.1).reshape(-1, 1)
plt.scatter(x, y, color='r')
plt.plot(x_grid, reg.predict(x_grid), color='b')
plt.title('Decision Tree Regression (DTR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# evaluate the model
r2 = r2_score(y, reg.predict(x))
print('R^2 score of the model:', r2)
