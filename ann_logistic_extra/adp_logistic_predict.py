import numpy as np
from adp_preprocess import get_binary_data

X, Y = get_binary_data()
# get the dimensions
D = X.shape[1]
W = np.random.randn(D) / np.sqrt(D)
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print(f"Score: {classification_rate(Y, predictions)}")
