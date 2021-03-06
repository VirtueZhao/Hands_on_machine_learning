import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn import datasets

# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

# plt.plot(X, y, 'bo')
# plt.show()

# X_b = np.c_[np.ones((100, 1)), X]
# print(np.c_[X])
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)

# X_new = np.array([[0], [2]])
# print(X_new)
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# print(X_new_b)
# y_predict = X_new_b.dot(theta_best)
# print(y_predict)

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.intercept_)
# print(lin_reg.coef_)
# print(lin_reg.predict(X_new))

# eta = 0.1
# n_iterations = 1000
# m = 100
#
# theta = np.random.randn(2, 1)
#
# for iteration in range(n_iterations):
#     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients

# n_epochs = 50
# t0, t1 = 5, 50

# print(theta_best)
# print([lin_reg.intercept_,lin_reg.coef_])
# print(theta)


# def learning_schedule(t):
#     return t0 / (t + t1)
#
#
# theta = np.random.randn(2, 1)
#
# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index: random_index+1]
#         yi = y[random_index: random_index+1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients

# print(theta)
# sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.intercept_, sgd_reg.coef_)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
# plt.plot(X, y, 'bo')
# plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
# print(X[0])
# print(X_poly[0])
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_, lin_reg.coef_)


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.show()


# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X, y)

# polynomial_regression = Pipeline((
#     ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
#     ("sgd_reg", LinearRegression()),
# ))
#
# plot_learning_curves(polynomial_regression, X, y)

# ridge_reg = Ridge(alpha=1, solver="cholesky")
# ridge_reg.fit(X, y)
# print(ridge_reg.predict([[1.5]]))
#
# sgd_reg = SGDRegressor(penalty="l2")
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))
#
# lasso_reg = Lasso(alpha=0.1)
# lasso_reg.fit(X, y)
# print(lasso_reg.predict([[1.5]]))
#
# sgd_reg = SGDRegressor(penalty="l1")
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))

# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net.fit(X, y)
# print(elastic_net.predict([[1.5]]))

iris = datasets.load_iris()
# print(list(iris.keys()))
# X = iris["data"][:,3:]
# y = (iris["target"] == 2).astype(np.int)
#
# log_reg = LogisticRegression()
# log_reg.fit(X, y)

# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g--", label="Iris_Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# plt.show()

# print(log_reg.predict([[1.7],[1.5]]))

X = iris["data"][:,(2,3)]
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X,y)

print(softmax_reg.predict([[5,2]]))
print(softmax_reg.predict_proba([[5,2]]))
