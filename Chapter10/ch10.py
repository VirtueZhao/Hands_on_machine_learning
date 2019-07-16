import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler

# iris = load_iris()
# X = iris.data[:,(2,3)]
# y = (iris.target == 0).astype(np.int)
#
# per_clf = Perceptron(random_state=42)
# per_clf.fit(X,y)
#
# y_pred = per_clf.predict([[2, 0.5]])
# print(y_pred)




#feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X)