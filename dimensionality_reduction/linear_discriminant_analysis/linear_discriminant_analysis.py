import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=0)

stc = StandardScaler()
x_train = stc.fit_transform(x_train)
x_test = stc.transform(x_test)

pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

clssifier = LogisticRegression()
clssifier.fit(x_train, y_train)

y_pred = clssifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix of the model is:', cm)
acc = accuracy_score(y_test, y_pred)
print('Accuracy of the model is:', acc)

x_set = x_train
y_set = y_train
x1, x2 = np.meshgrid(np.arange(min(x_set[:, 0]), max(x_set[:, 0]), step=0.01),
                 np.arange(min(x_set[:, 1]), max(x_set[:, 1]), step=0.01))

plt.contourf(x1, x2, clssifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'b', 'green')))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0],  x_set[y_set == j, 1], c=ListedColormap(['red', 'b', 'g'])(i), label=i)
plt.title('Principal component analysis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()