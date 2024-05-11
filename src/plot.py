import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import tensorflow.keras as keras


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


def plot_2d(x, y_list, y_legends=None):
    if not isinstance(y_list, list):
        y_list = [y_list]
    if y_legends is None:
        y_legends = ['y' + str(i) for i in range(len(y_list))]

    fig, ax = plt.subplots()
    for y, legend in zip(y_list, y_legends):
        ax.plot(x, y, label=legend)
    ax.legend()
    ax.grid()

    plt.show()


def scatter_2d(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ax.set(xlabel='x', ylabel='y')
    ax.grid()

    plt.show()


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


def dnn():
    x = tf.range(-2, 5, 0.01)
    y_true = tf.pow(x, 3) - 5 * tf.pow(x, 2) + 2 * x - 1
    y_true += tf.random.normal(shape=y_true.shape, mean=0, stddev=0.5)

    model = keras.Sequential(
        [
            keras.Input(shape=1),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1)
        ]
    )
    model.summary()

    batch_size = 10
    epochs = 200

    model.compile(loss=tf.losses.MSE, optimizer="adam")
    model.fit(x, y_true, batch_size=batch_size, epochs=epochs)

    y_pred = model(x)
    plot_2d(x, [y_true, y_pred], ['y_true', 'y_pred'])


def main():
    dnn()


if __name__ == "__main__":
    main()
