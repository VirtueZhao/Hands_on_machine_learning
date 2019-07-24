import tensorflow as tf
import numpy as np

n_inputs = 3
n_neurons = 5
#
# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
# Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons], dtype=tf.float32))
# b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))
#
# Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
# Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
#
# init = tf.global_variables_initializer()
#
# X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
# X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])
#
# with tf.Session() as sess:
#     init.run()
#     Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0:X0_batch, X1:X1_batch})
#
# print(Y0_val)
# print(Y1_val)

# X0 = tf.placeholder(tf.float32, [None, n_inputs])
# X1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1], dtype = tf.float32)
# Y0, Y1 = output_seqs
#
# init = tf.global_variables_initializer()
#
# X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
# X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])
#
# with tf.Session() as sess:
#     init.run()
#     Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0:X0_batch, X1:X1_batch})
#
# print(Y0_val)
# print(Y1_val)

# n_steps = 2
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
# basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
# outputs_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)
# outputs_seqs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
# outputs = tf.transpose(tf.stack(outputs_seqs), perm=[1, 0, 2])
#
# init = tf.global_variables_initializer()
#
# X_batch = np.array([
#     [[0, 1, 2], [9, 8, 7]],
#     [[3, 4, 5], [0, 0, 0]],
#     [[6, 7, 8], [6, 5, 4]],
#     [[9, 0, 1], [3, 2, 1]],
# ])
#
#
# with tf.Session() as sess:
#     init.run()
#     outputs_val = outputs.eval(feed_dict={X:X_batch})
#
# print(outputs_val)

seq_length = tf.placeholder(tf.int32,[None])



