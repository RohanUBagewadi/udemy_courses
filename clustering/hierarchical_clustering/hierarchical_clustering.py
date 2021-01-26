import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dataset = pd.read_csv('mall_customers.csv')
# only for visualization purpose column 3 and 4 are selected otherwise all columns should be selected
x = dataset.iloc[:, 1:].values

# encoding independent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = ct.fit_transform(x)

# plotting the dendrogram
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Observation points')
plt.ylabel('Euclidean distance')
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(x)

# visualizing only 2 features i.e x3 and x4
for i, j in enumerate(np.unique(y_hc)):
    plt.scatter(x[y_hc == j, 3], x[y_hc == j, 4],
                c=ListedColormap(('r', 'g', 'b', 'y', 'cyan'))(i), label=j)
plt.title('Hierarchical Clustering (Elbow-graph')
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.legend()
plt.show()
