import numpy as np
from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

# iris = datasets.load_iris()
# X = iris["data"][:,(2,3)]
# y = (iris["target"]==2).astype(np.float64)

# svm_clf = Pipeline((
#     ("scaler", StandardScaler()),
#     ("linear_svc", LinearSVC(C=1, loss="hinge"))
# ))
#
# svm_clf.fit(X, y)
# print(svm_clf.predict([[5.5, 1.7]]))

moons = make_moons()
X = moons[0]
y = moons[1]

# polynomial_svm_clf = Pipeline((
#     ("poly_features", PolynomialFeatures(degree=3)),
#     ("scaler", StandardScaler()),
#     ("svm_clf", LinearSVC(C=10, loss="hinge"))
# ))
#
# polynomial_svm_clf.fit(X,y)

# poly_kernel_svm_clf = Pipeline((
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
# ))
# poly_kernel_svm_clf.fit(X,y)

# rbf_kernel_svm_clf = Pipeline((
#     ("scaler", StandardScaler()),
#     ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
# ))
# rbf_kernel_svm_clf.fit(X,y)

# svm_reg = LinearSVR(epsilon=1.5)
# svm_reg.fit(X,y)

# svm_poly_reg = SVR(kernel="poly",degree=2,C=100,epsilon=0.1)
# svm_poly_reg.fit(X,y)