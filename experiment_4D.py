import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class LogisticRegression():
  def loss(self, X, y, theta):
    YX = X * y[:, None]
    hy = YX @ theta
    loss = np.log(1 + np.exp(-hy)).mean()
    error = (hy <= 0).mean()
    return loss, error

  def gradient(self, X, y, theta):
    YX = X * y[:, None]
    m = X.shape[0]
    gradient = -YX.T @ (1 / (1 + np.exp(YX @ theta))) / m
    return gradient

  def gradient_descent(self, X, y, alpha, iters):
    """
    
    :param X: 
    :param y: 
    :param alpha: learning rate  
    :param iters: 
    :return: 
    """
    m, n = X.shape
    theta = np.zeros(n)
    loss, err = np.zeros(iters), np.zeros(iters)
    grads = np.zeros((iters, X.shape[1]))

    for t in range(iters):
      loss[t], err[t] = self.loss(X, y, theta)
      # print(self.gradient(X, y, theta))

      grads[t] = self.gradient(X, y, theta)
      theta -= alpha * grads[t]
      # print("At step {}: theta={}, loss={}, err={}, grad={}".
      #       format(t, theta, loss[t], err[t], grads[t]))
    return theta, loss, err, grads


if __name__ == "__main__":
  # LOAD DATA
  df = pd.read_table('data/iris.data.txt', sep=',',
                     names=['sepal length', 'sepal width', 'petal length',
                            'petal width', 'label'])
  df = df[
    df['label'].isin(['Iris-setosa', 'Iris-versicolor'])]  # only keep 2 classes
  df = df.reset_index(drop=True)
  print('Dimensions:', df.shape)
  df.head()



  # post - processing
  X = df.drop('label', axis=1)
  X.columns = ['X1', 'X2', 'X3', 'X4']
  y = df['label']
  print(X.shape, y.shape)



  # # PLOT
  df = pd.DataFrame(X, columns=['X1', 'X2', 'X3', 'X4'])
  df['label'] = y
  df.head()

  # plt.figure(figsize=(8, 5))
  # plt.scatter(df['X1'][df['label'] == 'Iris-setosa'],
  #             df['X3'][df['label'] == 'Iris-setosa'],
  #             marker='x', color='C0')
  # plt.scatter(df['X1'][df['label'] == 'Iris-versicolor'],
  #             df['X3'][df['label'] == 'Iris-versicolor'],
  #             marker='+',
  #             color='C3')
  # plt.title('Data Dimensions')
  # plt.xlabel('X1')
  # plt.ylabel('X2')
  # plt.show()


  # MODEL
  X = df[['X1', 'X2', 'X3', 'X4']].as_matrix()
  # X = np.array(X, dtype=np.float32)
  # convert to 1 / -1 labels
  y = ((df['label'] == 'Iris-setosa') * 2 - 1).as_matrix()
  y = np.array(y, dtype=np.int)

  # define a model
  n_epochs = 5000
  model = LogisticRegression()
  alpha = 0.1
  theta, loss, err, grads = model.gradient_descent(X, y, alpha, n_epochs)
  print("Converged theta = {}".format(theta))

  # # PLOT
  # plt.figure(figsize=(8, 5))
  # plt.scatter(df['X1'][df['label'] == 'Iris-setosa'],
  #             df['X2'][df['label'] == 'Iris-setosa'], marker='x', color='C0')
  # plt.scatter(df['X1'][df['label'] == 'Iris-versicolor'],
  #             df['X2'][df['label'] == 'Iris-versicolor'], marker='+',
  #             color='C3')
  # line_x2_points = np.linspace(-5, 5)
  # line_x1_points = -theta[1] / theta[0] * line_x2_points
  # plt.plot(line_x1_points, line_x2_points, 'k-')
  # plt.title('Logistic Regression Classifier')
  # plt.xlabel('X1')
  # plt.ylabel('X2')
  # plt.show()

  # VISUALIZING LOSS
  plt.figure(figsize=(8, 5))
  epochs = range(n_epochs)
  plt.close()
  plt.plot(epochs, loss)
  plt.title('Loss over Epochs')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()

  # Creating a grid of theta values (centered at optimal theta)
  # for evaluating loss.
  theta_0_range = np.linspace(theta[0] - 5, theta[0] + 5, 50)
  theta_1_range = np.linspace(theta[1] - 5, theta[1] + 5, 50)
  theta_2_range = np.linspace(theta[2] - 5, theta[2] + 5, 50)
  theta_3_range = np.linspace(theta[3] - 5, theta[3] + 5, 50)

  theta_0, theta_1 = np.meshgrid(theta_0_range, theta_1_range)
  theta_2, theta_3 = np.meshgrid(theta_2_range, theta_3_range)

  plt.close()
  plt.plot(theta_0, theta_1, '.', color='k')
  plt.xlabel('theta0')
  plt.ylabel('theta1')
  plt.show()

  # Visualizing the loss landscape with a contour plot.
  grid_rows, grid_cols = theta_0.shape
  loss = np.zeros((grid_rows, grid_cols))
  for i in range(grid_rows):
    for j in range(grid_cols):
      theta_0_ij = theta_0[i, j]
      theta_1_ij = theta_1[i, j]
      theta_2_ij = theta_2[i, j]
      theta_3_ij = theta_3[i, j]
      theta = np.array([theta_0_ij, theta_1_ij, theta_2_ij, theta_3_ij])
      loss[i, j] = model.loss(X, y, theta)[0]

  # perform gradient
  n_step = 500
  theta_ = [np.array((0., 0., 0., 0.))]
  loss_ = [model.loss(X, y, theta_[-1])[0]]  # also return error
  for j in range(n_step-1):
    prev_theta = theta_[-1]

    # update now based on the closed form formula
    current_grad = model.gradient(X, y, prev_theta)
    current_theta = prev_theta - alpha * current_grad

    theta_.append(current_theta)
    loss_.append(model.loss(X, y, theta_[-1])[0])
    # print("Step {}, prev_theta={}, current_theta={}, loss={}".
    #       format(j, prev_theta, current_theta, loss_[-1]))

  plt.figure(figsize=(8, 5))
  contour = plt.contour(theta_0, theta_1, loss)


  # narrow down theta_ to 2D for plotting
  selected_idx = [0, 1]
  theta = [(x[selected_idx[0]], x[selected_idx[1]]) for x in theta_]

  # draw gradient
  for j in range(1, n_step):
    plt.annotate('', xy=theta[j],
                 xytext=theta[j-1],
                 arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                 va='center', ha='center')

  # more decoration
  colors = ['b', 'g', 'm', 'c', 'orange']
  plt.close()
  plt.scatter(*zip(*theta), c=colors, s=40, lw=0)

  plt.clabel(contour, inline=1, fontsize=10)
  plt.title('Loss over parameters: contour plot')
  plt.xlabel('theta0')
  plt.ylabel('theta1')
  plt.show()


  # 3D plot
  from mpl_toolkits import mplot3d
  from mpl_toolkits.mplot3d import Axes3D, proj3d
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

  fig = plt.figure(figsize=(12, 7))
  ax = plt.axes(projection='3d')
  ax.plot_surface(theta_0, theta_1, loss)

  # draw gradient
  for j in range(1, n_step):
    a = Arrow3D([theta[j-1][0], theta[j][0]],
                [theta[j-1][1], theta[j][1]],
                [loss_[j-1], loss_[j]],
                mutation_scale=20,
                lw=1, arrowstyle="-|>", color='r')
    ax.add_artist(a)

  # more decoration
  first_theta_, second_theta_ = zip(*theta)
  ax.scatter(first_theta_, second_theta_, loss_,
             c=colors*100,
             s=40, lw=0,
             zorder=1)


  ax.set_title('Loss landscape')
  ax.set_xlabel('theta0')
  ax.set_ylabel('theta1')
  ax.set_zlabel('Loss')
  ax.view_init(40, 250)

  plt.draw()
  plt.show()