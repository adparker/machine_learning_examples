import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_binary_data

Xtrain, Ytrain, Xtest, Ytest = get_binary_data()
#X, Y = shuffle(X, Y)

# Separate out test and training data.
# testsize = 100
# Xtrain = X[:-testsize]
# Ytrain = Y[:-testsize]
# Xtest = X[-testsize:]
# Ytest = Y[-testsize:]

# Randomly initialize weights
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y==P)

def cross_entropy(T, pY):
    return -np.mean(T * np.log(pY) + (1-T) * np.log(1-pY))

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    # Make the prediction
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    # Calculate the cost
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    # Do one step of gradient descent
    W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate * (pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(f"{i, ctrain, ctest}")

print(f"Final train classification rate: {classification_rate(Ytrain, np.round(pYtrain))}")
print(f"Final test classification rate: {classification_rate(Ytest, np.round(pYtest))}")

legend1, = plt.plot(train_costs, label='train costs')
legend2, = plt.plot(test_costs, label='test costs')
plt.legend([legend1, legend2])
plt.show()
