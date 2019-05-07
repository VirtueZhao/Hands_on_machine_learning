import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# plt.plot(X, y, 'bo')
# plt.show()

X_b = np.c_[np.ones((100, 1)), X]
# print(np.c_[X])
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)

X_new = np.array([[0], [2]])
# print(X_new)
X_new_b = np.c_[np.ones((2, 1)), X_new]
# print(X_new_b)
y_predict = X_new_b.dot(theta_best)
# print(y_predict)

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
# print(lin_reg.intercept_)
# print(lin_reg.coef_)
# print(lin_reg.predict(X_new))

eta = 0.1
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

n_epochs = 50
t0, t1 = 5, 50

# print(theta_best)
# print([lin_reg.intercept_,lin_reg.coef_])
# print(theta)


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index: random_index+1]
        yi = y[random_index: random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

# print(theta)
sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)



