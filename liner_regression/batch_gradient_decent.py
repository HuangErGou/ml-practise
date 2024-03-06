import sys

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

from helper import general
from helper import linear_regression as lr

sys.path.append('..')

data = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
print(data.head)

x = general.get_x(data)
print(x.shape, type(x))

y = general.get_y(data)
print(y.shape, type(y))

theta = np.zeros(x.shape[1])
cost = lr.cost(theta, x, y)
print(cost)

epoch = 500
final_theta, cost_data = lr.batch_gradient_decent(theta, x, y, epoch, alpha=0.01)

print(lr.cost(final_theta, x, y))
print(len(cost_data))
# print(np.arange(1, len(cost_data) + 1))

line_plot_data = DataFrame({'x': np.arange(1, len(cost_data) + 1), 'cost': cost_data})

plt.subplot(1, 2, 1)
ax = sns.lineplot(line_plot_data, x='x', y='cost')
ax.set_xlabel('epoch')
ax.set_ylabel('cost')

b = final_theta[0]
m = final_theta[1]
plt.subplot(1, 2, 2)
sns.scatterplot(data, x='population', y='profit')
plt.plot(data.population, data.population * m + b)
plt.show()
