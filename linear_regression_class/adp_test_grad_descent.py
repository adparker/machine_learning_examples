import numpy as np
import matplotlib.pyplot as plt

N = 900
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N) # add noise
#Y[-1] += 30
#Y[-2] += 30
alpha = 1e-3

# plt.scatter(X, Y)
# plt.show()

# Add the bias term
X = np.vstack([np.ones(N), X]).T

def gradient_descent(X,y,alpha,num_iters):
    num_features = X.shape[1]
    theta = np.zeros(num_features)
    for n in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = X.T.dot(errors) / len(y) * 50
        #gradient = X.T.dot(errors)
        #step = np.maximum([-0.1,-0.1], np.minimum([0.1, 0.1], alpha * gradient))
        step = alpha * gradient
        theta -= step

    return theta

theta = gradient_descent(X, Y, alpha, 1000)
print(theta)
Yhat = X.dot(theta)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat, label='prediction')
plt.legend()
plt.show()
