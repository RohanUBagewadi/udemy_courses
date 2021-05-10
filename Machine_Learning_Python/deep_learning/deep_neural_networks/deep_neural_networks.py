import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('churn_modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = ct.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=0)

stc = StandardScaler()
x_train = stc.fit_transform(x_train)
x_test = stc.transform(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(6, 'relu'))
model.add(tf.keras.layers.Dense(10, 'relu'))
model.add(tf.keras.layers.Dense(1, 'sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=1)
# model.summary()

new_x = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 5000]])

new_x[:, 2] = le.transform(new_x[:, 2])
new_x = ct.transform(new_x)

new_x = stc.transform(new_x)

y_pred = model.predict(new_x)

y_pred[y_pred >= 0.5] = 1
print(y_pred)

y_pred = model.predict(x_test)
y_pred = (y_pred >= 0.5).astype('int')

cm = confusion_matrix(y_test, y_pred.reshape(-1))
acc = accuracy_score(y_test, y_pred)
print('Confusion matrix of model:', cm)
print('Accuracy of model:', acc)
