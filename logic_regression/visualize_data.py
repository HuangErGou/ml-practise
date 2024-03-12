import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
print(df.shape)
print(df.head())
print(df.describe())

sns.lmplot(x='exam1', y='exam2', hue='admitted', data=df, fit_reg=False, scatter_kws={'s': 50})
plt.show()
