import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing dataset
dataset = pd.read_csv('data_preprocessing/data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# encoding categorical data
# encoding independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = ct.fit_transform(x)

# encoding dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# feature scaling (standardisation works for most of the cases) normalization for data distributed normally
sts = StandardScaler()
x_train[:, 3:] = sts.fit_transform(x_train[:, 3:])     # no standardization for categorical columns i.e dummy variables
x_test[:, 3:] = sts.transform(x_test[:, 3:])

