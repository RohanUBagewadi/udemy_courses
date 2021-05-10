import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv('salary_data.csv')
x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)

# prediction
pred = lr.predict(x_test)
error = y_test - pred

plt.scatter(x_train, y_train, color='r')
plt.plot(x_train, lr.predict(x_train), color='b')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# evaluate the model
r2 = r2_score(y_test, lr.predict(x_test))
print('R^2 score of the model:', r2)

