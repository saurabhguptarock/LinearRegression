import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dfx = pd.read_csv('weightedX.csv')
dfy = pd.read_csv('weightedY.csv')

x = dfx.values
y = dfy.values

x = x.reshape((-1,))
y = y.reshape((-1,))

X = (x - x.mean()) / x.std()
Y = y

x1 = X
x2 = X**2
X = np.stack((x1, x2), axis=1)

model = LinearRegression()
model.fit(X, Y)
output = model.predict(X)

plt.scatter(X[:, 0], Y)
plt.scatter(X[:, 0], output, color='orange')
plt.show()
