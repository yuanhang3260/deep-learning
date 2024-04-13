import tensorflow as tf


def comp_conv2d(conv2d, x):
    # 这里的（1，1）表示批量大小和通道数都是1
    x = tf.reshape(x, (1, ) + x.shape + (1, ))
    y = conv2d(x)
    # 省略前两个维度：批量大小和通道
    return tf.reshape(y, y.shape[1:3])


def main():
    conv2d = tf.keras.layers.Conv2D(filters=1, kernel_size=2, kernel_initializer=tf.initializers.random_normal)

    x = tf.constant(value=range(24), shape=[1, 3, 4, 2], dtype=float)
    y = conv2d(x)
    print(y)


if __name__ == "__main__":
    main()

