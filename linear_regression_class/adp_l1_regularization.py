import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50
X = (np.random.random((N,D)) - 0.5)*10
true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))
Y = X.dot(true_w) + np.random.randn(N) * 0.5

# now do gradient descent
costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 10

for i in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    gradient = X.T.dot(delta) + l1 * np.sign(w)
    w = w - learning_rate * gradient
    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()
print(f"final w:{w}")

plt.plot(true_w, label='true w')
plt.plot(w, label = 'map w')
plt.legend()
plt.show()