import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt
from helper import general as general
from helper import logic_regression as lr

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
print(data.head())

x = general.get_x(data)
print(x.shape)

y = general.get_y(data)
print(y.shape)

theta = np.zeros(3)
print(theta)

print(lr.cost(theta, x, y))
print(lr.gradient(theta, x, y))

res = opt.minimize(fun=lr.cost, x0=theta, args=(x, y), method='Newton-CG', jac=lr.gradient)
print(type(res))
print(res)

final_theta = res.x
print(res.x)

# https://www.cnblogs.com/volcao/p/9368030.html
# θT.xb = θ0 + θ1.x1 + θ2.x2 = 0，则该边界是一条直线，因为分类问题中特征空间的坐标轴都表示特征；
# 那么 x2 可以直接算出来

coef = -(res.x / res.x[2])
print(coef)
x = np.arange(130, step=0.1)
y = coef[0] + coef[1] * x
print(y)

sns.lmplot(data=data, x='exam1', y='exam2', hue='admitted', fit_reg=False, scatter_kws={"s": 25})
plt.plot(x, y)
plt.xlim(0, 130)
plt.ylim(0, 130)
plt.show()
