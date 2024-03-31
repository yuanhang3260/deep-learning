import tensorflow as tf


class Model:
    @property
    def trainable_variables(self):
        raise NotImplementedError()


def softmax(o):
    o_exp = tf.exp(o)
    partition = tf.reduce_sum(o_exp, axis=1, keepdims=True)
    return o_exp / partition


def cross_entropy(y, y_hat):
    loss = -tf.math.log(
        tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
    )
    return tf.reduce_sum(loss) / loss.shape[0]

