import tensorflow as tf

def main():
    a = tf.Variable(tf.constant([[1., 1., 1.], [2., 2., 2.]]))
    b = tf.Variable(tf.constant([[1., 2., 3.], [4., 5., 6.]]))
    with tf.GradientTape() as g:
        c = (a * b)

    print(c)

    da, db = g.gradient(c, sources=[a, b])
    print(da)
    print(db)


if __name__ == "__main__":
    main()