import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

dataset = pd.read_csv('mall_customers.csv')
# only for visualization purpose column 3 and 4 are selected otherwise all columns should be selected
x = dataset.iloc[:, [3, 4]].values

# encoding independent variable
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# x = ct.fit_transform(x)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('K-Means Clustering (Elbow-graph')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(x)  # returns the dependent variable i.e the centroids

for i, j in enumerate(np.unique(y_kmeans)):
    plt.scatter(x[y_kmeans == j, 0], x[y_kmeans == j, 1],
                c=ListedColormap(('r', 'g', 'b', 'y', 'cyan'))(i), label=j)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c='m', label='Centroids')
plt.title('K-Means Clustering (Elbow-graph')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.legend()
plt.show()
