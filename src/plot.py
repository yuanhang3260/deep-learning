import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf


def synthetic_data(w, b, num_examples):
    x = tf.random.normal(shape=(num_examples, len(w)), mean=0, stddev=1)
    y = tf.math.sigmoid(tf.matmul(x, w) + b)
    y += tf.random.normal(shape=y.shape, mean=0, stddev=0.1)
    return x, y


def linear():
    true_w = tf.constant([2, -3.4], shape=(2, 1))
    true_b = 4.2
    x, y = synthetic_data(true_w, true_b, 1000)

    w1 = tf.range(-20, 20, 0.1)
    w2 = tf.range(-20, 20, 0.1)
    w1, w2 = tf.meshgrid(w1, w2)
    w = tf.Variable([tf.reshape(w1, shape=(-1)), tf.reshape(w2, shape=(-1))])

    #scatter_2d(x[:, 0], y)
    y_hat = tf.math.sigmoid(tf.matmul(x, w) + true_b)
    loss = tf.reduce_mean(tf.square(y_hat - y), axis=0)
    plot_3d_surface(w1, w2, tf.reshape(loss, shape=(len(w1), len(w2))))


def plot_3d_surface(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    ax.grid(True, which='both')
    ax.contour3D(x, y, z, zdir='z', offset=-5, cmap=cm.coolwarm)

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    #ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()


def scatter_3d(x, y, z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    ax.scatter(x, y, z, marker='o')

    #ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()


def plot_2d(x, y):
    #x = np.arange(0.0, 1.0, 0.01)
    #y = x ** 0.7 + (1 - x) ** 0.3
    #y = 0.7 * np.log(x) + 0.3 * np.log(1 - x)

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='x', ylabel='y')
    ax.grid()

    plt.show()


def scatter_2d(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ax.set(xlabel='x', ylabel='y')
    ax.grid()

    plt.show()


def main():
    linear()


if __name__ == "__main__":
    main()
