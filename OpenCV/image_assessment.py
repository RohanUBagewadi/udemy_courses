import numpy as np
import PIL
import matplotlib.pyplot as plt

np.random.seed(101)

a = np.full(fill_value=10, shape=(5,5))
arr = np.random.randint(0, 100, (5, 5))
print('max value:', arr.max())
print('max value:', arr.min())

