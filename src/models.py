import tensorflow as tf


class Model:
    @property
    def trainable_variables(self):
        raise NotImplementedError()


def softmax(x):
    x_exp = tf.exp(x)
    partition = tf.reduce_sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


def cross_entropy(y, y_hat):
    loss = -tf.math.log(
        tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
    )
    return tf.reduce_sum(loss) / loss.shape[0]

