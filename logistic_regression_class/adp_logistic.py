import numpy as np
N=100
D=2

X = np.random.randn(N, D)
Xb = np.vstack((np.ones(N), X.T)).T
Xb = np.hstack((np.ones((N,1)), X))
w = np.random.randn(D+1)
z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
