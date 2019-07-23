import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

# iris = load_iris()
# X = iris.data[:,(2,3)]
# y = (iris.target == 0).astype(np.int)
#
# per_clf = Perceptron(random_state=42)
# per_clf.fit(X,y)
#
# y_pred = per_clf.predict([[2, 0.5]])
# print(y_pred)


# mnist = fetch_mldata('MNIST original')
# X, y = mnist["data"], mnist["target"]
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# shuffle_index = np.random.permutation(60000)
# X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# y_train = y_train.astype(int)
#
# feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
# dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300,100], n_classes=10, feature_columns=feature_columns)
# dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)
# y_pred = list(dnn_clf.predict(X_test))
# print(accuracy_score(y_test, y_pred))
# print(dnn_clf.evaluate(X_test,y_test))

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.compat.v1.placeholder(tf.int64, shape=None, name="y")


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


with tf.name_scope("dnn"):
    # hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    # hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    # logits = neuron_layer(hidden2, n_outputs, "outputs")
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()
print("Construction Phase Done")

mnist = input_data.read_data_sets("/tmp/data/")

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_exaples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})

        print(epoch, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)

    save_path = saver.save(sess, "./my_model_final.ckpt")