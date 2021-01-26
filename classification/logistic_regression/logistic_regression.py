import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('social_network_ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

stc = StandardScaler()
x_train = stc.fit_transform(x_train)
x_test = stc.transform(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('Confusion matrix of model:', cm)
print('Accuracy of model:', acc)

from matplotlib.colors import ListedColormap

x_set, y_set = stc.inverse_transform(x_train), y_train
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 10, x_set[:, 0].max() + 10, step=0.25),
                     np.arange(x_set[:, 1].min() - 1000, x_set[:, 1].max() + 1000, step=0.25))

plt.contourf(x1, x2, classifier.predict(stc.transform(np.array([x1.ravel(), x2.ravel()]).T)).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (training set)')
plt.xlabel('Ages')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
