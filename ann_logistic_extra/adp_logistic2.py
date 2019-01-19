import numpy as np
import matplotlib.pyplot as plt
N = 100
D = 2
X = np.random.randn(N, D)

# First 50 centered at -2, -2
X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

# Create an array of targets, first 50 to 0, second 50 to 1.
T = np.concatenate((np.zeros((round(N/2), 1)), np.ones((round(N/2), 1))))
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)
#T = np.array([0]*50 + [1]*50)

# add a column of ones
# ones = np.array([[1]*N]).T
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# Initialize weights - [w0, w1, w2]
w = np.random.randn(D + 1)

# Calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    # T * -ln(Y) + (1 - T)*-ln(1-Y)
    E = 0
    for i in range(len(T)):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print(f"{cross_entropy(T, Y)}")
w = np.array([0, 4, 4])
z = Xb.dot(w)
Y = sigmoid(z)
print(f"{cross_entropy(T, Y)}")

plt.scatter(X[:, 0], X[:,1], c=T[:,0], s=100, alpha=0.5)
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()
