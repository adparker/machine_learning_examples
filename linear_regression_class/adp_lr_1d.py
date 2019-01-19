import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data_1d.csv", header=None)
df.columns = ['x', 'y']

df.plot.scatter(x='x', y='y')
plt.show()

# apply the equations
x = df.x + 1000
y = df.y + 1000
denominator = x.dot(x) - x.mean() * x.sum()
a = (x.dot(y) - y.mean()*x.sum()) / denominator
b = (y.mean() * x.dot(x) - x.mean() * x.dot(y)) / denominator

# calculate predicted y
yhat = a*x + b

# plot it all
plt.scatter(x, y)
plt.plot(x, yhat)
plt.show()

# calculate r-squared
d1 = y - yhat
d2 = y - y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print(f"Rsquared is {r2}")

