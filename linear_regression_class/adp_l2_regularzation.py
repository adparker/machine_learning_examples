import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N) # add noise
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# Add the bias term
X = np.vstack([np.ones(N), X]).T
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) # ML version
#w_ml = scipy.linalg.solve(X.T.dot(X), X.T.dot(Y)) # ML version

Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.scatter(X[:,1], Yhat_ml)
plt.show()

l2 = 1000.0
#w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y)) # MAP version
w_map = np.linalg.solve(np.array([[l2, 0], [0, l2]]) + X.T.dot(X), X.T.dot(Y)) # MAP version
Yhat_map = X.dot(w_map)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=l2, solver='cholesky', fit_intercept=False)
X_no_intercept = np.reshape(X[:, 1], (-1, 1))
#clf.fit(X_no_intercept, Y)
clf.fit(X, Y)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label='map')
#plt.plot(X[:,1], clf.predict(X_no_intercept), label='sklearn_ridge')
plt.plot(X[:,1], clf.predict(X), label='sklearn_ridge')

plt.legend()
plt.show()

###
import numpy as np
A = np.asmatrix(np.random.rand(10,1))
A = np.vstack([np.ones(N), A]).T

b = np.asmatrix(np.random.rand(10,1))
I = np.identity(A.shape[1])
alpha = 100
#x = np.linalg.inv(A.T*A + alpha * I)*A.T*b
x = np.linalg.solve(A.T.dot(A) + alpha * I, A.T.dot(b))
print (x.T)


from sklearn.linear_model import Ridge
model = Ridge(alpha = alpha, tol=0.1, fit_intercept=False, solver='cholesky').fit(A ,b)

print(model.coef_)
print (model.intercept_)
