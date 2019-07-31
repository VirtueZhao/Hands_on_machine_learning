import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
# l2_reg = 0.001
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# with tf.contrib.framework.arg_scope(
#     [fully_connected],
#     activation_fn=tf.nn.elu,
#     weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
#     weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)):
#     hidden1 = fully_connected(X, n_hidden1)
#     hidden2 = fully_connected(hidden1, n_hidden2)
#     hidden3 = fully_connected(hidden2. n_hidden3)
#     outputs = fully_connected(hidden3, n_outputs, activation_fn=None)
#
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
#
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# loss = tf.add_n([reconstruction_loss] + reg_losses)
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# n_epochs = 5
# batch_size = 150
#
# mnist = input_data.read_data_sets("/tmp/data")
#
# with tf.Seesion() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001

activation = tf.nn.elu
regularizer = tf.contrib.layer.l2_regularizer(l2_reg)
initializer = tf.contrib.layer.variance_scaling_initializer()

X = tf.placeholder(tf.float, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name="weights3")
weights4 = tf.transpose(weights1, name="weights4")

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variable_initializer()