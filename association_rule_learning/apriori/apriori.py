import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('market_basket_optimisation.csv')

x = dataset.iloc[:, :].values
t = np.ndarray.tolist(x)
transactions = []
for i in range(0, x.shape[0]):
    transactions.append([j for j in dataset.iloc[i, :].values if str(j) != 'nan'])

rules = apriori(transactions=t, min_support=0.003,
                min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
print([rules])