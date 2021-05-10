import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('social_network_ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

stc = StandardScaler()
x_train = stc.fit_transform(x_train)
x_test = stc.transform(x_test)

# using linear kernel
classifier = SVC(kernel='rbf')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('Confusion matrix of model:', cm)
print('Accuracy of model:', acc)

accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10)
print('Accuracy: {:0.2f} %'.format(accuracies.mean()*100))
print('Standard division: {:0.2f} %'.format(accuracies.std()*100))

parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10,
                           n_jobs=-1)
grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print('Best accuracy: {:0.2f} %'.format(best_accuracy*100))
print('Standard division: {}'.format(best_parameters))

