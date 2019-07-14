import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import normalize
from datetime import datetime

#print(tf.__version__)

# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")
# f = x*x*y + y + 2
#
# sess = tf.compat.v1.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()
#
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
#     print(result)
#
# init = tf.compat.v1.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     result = f.eval()
#     print(result)
#
# sess = tf.compat.v1.InteractiveSession()
# init.run()
# result = f.eval()
# print(result)
# sess.close()

# x1 = tf.Variable(1)
# # print(x1.graph is tf.compat.v1.get_default_graph())
# #
# # graph = tf.Graph()
# # with graph.as_default():
# #     x2 = tf.Variable(2)
# #
# # print(x2.graph is graph)
# # print(x2.graph is tf.compat.v1.get_default_graph())

# w = tf.constant(3)
# x = w + 2
# y = x + 5
# z = x * 3
#
# with tf.Session() as sess:
#     print(y.eval())
#     print(z.eval())
#
# with tf.Session() as sess:
#     y_val, z_val = sess.run([y,z])
#     print(y_val)
#     print(z_val)

# housing = fetch_california_housing()
# m, n = housing.data.shape
# # print(housing.data.shape)
# housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
#
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)
#
# with tf.Session() as sess:
#     theta_value = theta.eval()
#     print(theta_value)

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir,now)

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]
scaled_housing_data_plus_bias = normalize(housing_data_plus_bias)

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32,name="y")
theta = tf.Variable(tf.random.uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
print(error.op.name)
print(mse.op.name)
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()


file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
            save_path = saver.save(sess, "./tmp/my_model.ckpt")
            mse_summary = tf.compat.v1.summary.scalar('MSE', mse)
            # file_writer.add_summary(mse_summary)
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, "./temp/my_model_final.ckpt")
    file_writer.close()

# A = tf.placeholder(tf.float32, shape=(None,3))
# B = A + 5
# with tf.Session() as sess:
#     B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
#     B_val_2 = B.eval(feed_dict={A:[[4,5,6],[7,8,9]]})
#
# print(B_val_1)
# print(B_val_2)

