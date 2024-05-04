import tensorflow as tf


def test_grad():
    a = tf.Variable(tf.constant([[1., 1., 1.], [2., 2., 2.]]))
    b = tf.Variable(tf.constant([[1., 2., 3.], [4., 5., 6.]]))
    with tf.GradientTape() as tape:
        c = (a * b)

    print(c)

    da, db = tape.gradient(c, sources=[a, b])
    print(da)
    print(db)

@tf.function
def f(x):
    # a = tf.constant([[1, 2], [3., 4]])
    # x = tf.constant([[1., 0.], [0., 1.]])
    # # b = tf.Variable(12.)
    # y = tf.matmul(a, x)
    #
    # print("PRINT: ", y)
    # tf.print("TF-PRINT: ", y)
    with tf.GradientTape() as tape:
        y = x * 2
    grad = tape.gradient(y, x)
    return y, grad


def main():
    print('--------1----------')
    x = tf.Variable(tf.constant([1., 2.]))
    y, grad = f(x)
    print(grad)
    print('--------2----------')

    print(tf.autograph.to_code(f.python_function))

    test_grad()


if __name__ == "__main__":
    main()
