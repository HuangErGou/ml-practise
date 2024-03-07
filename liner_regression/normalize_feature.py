import numpy as np
import pandas as pd
import sys
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

sys.path.append('..')

from helper import linear_regression as lr
from helper import general as general

raw_data = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])
print(raw_data.head())

data = general.normalize_feature(raw_data)
print(data.head())

x = general.get_x(data)
print(x.shape, type(x))

y = general.get_y(data)
print(y.shape, type(y))

alpha = 0.01
theta = np.zeros(x.shape[1])
epoch = 500

final_theta, cost_data = lr.batch_gradient_decent(theta, x, y, epoch, alpha=alpha)

line_plot_data = DataFrame({'x': np.arange(1, len(cost_data) + 1), 'cost': cost_data})
ax = sns.lineplot(line_plot_data, x='x', y='cost')
plt.show()
