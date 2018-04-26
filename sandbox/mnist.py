from __future__ import print_function


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

MAX_STEPS = 500

import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow as tf


from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
  def __init__(self, xs, ys, zs, *args, **kwargs):
    FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
    self._verts3d = xs, ys, zs

  def draw(self, renderer):
    xs3d, ys3d, zs3d = self._verts3d
    xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
    self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
    FancyArrowPatch.draw(self, renderer)


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

all_trainable_params = tf.trainable_variables()

losses = []
params = []

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  max_steps = MAX_STEPS
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    loss, _, param  = sess.run([cross_entropy, train_step, all_trainable_params],
                                feed_dict={x: batch_xs,y_: batch_ys})

    losses.append(loss)
    params.append(param)

    # if (step % 100) == 0:
    #   print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

  print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# get the final values
converged_W, converged_b = params[-1][0], params[-1][1]


# get the delta changes over all steps
delta_W = np.zeros((784, 10))
for i in range(1, len(params)):
  delta_W += np.abs(params[i-1][0] - params[i][0])


# get the 2 max location
first_location = np.unravel_index(np.argmax(delta_W, axis=None), delta_W.shape)
delta_W[first_location] = 0
second_location = np.unravel_index(np.argmax(delta_W, axis=None), delta_W.shape)
print("Two thetas changed the most: {}, {}".format(first_location, second_location))


# record the to-be-changed axes
theta_1_converged = converged_W[first_location]
theta_2_converged = converged_W[second_location]


# calculate the losses with these 2 axis changing around its convered values
n_steps = 1000
theta_1_range = np.linspace(theta_1_converged-500, theta_1_converged+500,
                            n_steps)
theta_2_range = np.linspace(theta_2_converged-500, theta_2_converged+500,
                            n_steps)

# new graphs
batch_size=1000
x = tf.placeholder(tf.float32, [batch_size, 784])
weight = tf.placeholder(tf.float32, [784, 10])  # converged_W
bias = tf.placeholder(tf.float32, [10])  # converged_bias
y = tf.nn.softmax(tf.matmul(x, weight) + bias)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)),
                                              reduction_indices=[1]))


losses_for_plot = []
with tf.Session() as sess:
  # we only use fixed data
  batch_xs, batch_ys = mnist.train.next_batch(1000)

  # now recording the loss and values of those points
  for i in range(n_steps):
    current_theta_1 = theta_1_range[i]
    current_theta_2 = theta_2_range[i]

    # update repective values in converged W
    converged_W[first_location] = current_theta_1
    converged_W[second_location] = current_theta_2

    # now get the loss
    loss = sess.run(cross_entropy,
                    feed_dict={x: batch_xs,
                               y_: batch_ys,
                               weight: converged_W,
                               bias: converged_b
                               })
    print(current_theta_1, current_theta_2, loss)
    losses_for_plot.append([current_theta_1, current_theta_2, loss])


# dump data to disk
with open('data.pkl', 'wb') as f:
  pickle.dump(np.array(losses_for_plot), f, pickle.HIGHEST_PROTOCOL)

# draw now
colors = ['b', 'g', 'm', 'c', 'orange']

theta_01, theta_10 = np.meshgrid(theta_1_range, theta_2_range)
grid_rows, grid_cols = theta_01.shape
viz_losses = np.zeros((grid_rows, grid_cols))
for i in range(grid_rows):
  for j in range(grid_cols):
    viz_losses[i, j] = losses_for_plot[i][2]


fig = plt.figure(figsize=(12, 7))
ax = plt.axes(projection='3d')

ax.plot_surface(theta_01, theta_10, viz_losses)

# # draw gradient
# for j in range(1, MAX_STEPS):
#   a = Arrow3D([theta[j - 1][0], theta[j][0]],
#               [theta[j - 1][1], theta[j][1]],
#               [losses[j - 1], losses[j]],
#               mutation_scale=20,
#               lw=1, arrowstyle="-|>", color='r')
#   ax.add_artist(a)

plt.draw()
plt.show()














# print(__doc__)
#
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_mldata
# from sklearn.neural_network import MLPClassifier
#
# mnist = fetch_mldata("MNIST original")
# # rescale the data, use the traditional train/test split
# X, y = mnist.data / 255., mnist.target
# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]
#
# # mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
# #                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
# mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1)
#
# mlp.fit(X_train, y_train)
# print("Training set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))

# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
#
# import pdb; pdb.set_trace()
# plt.show()