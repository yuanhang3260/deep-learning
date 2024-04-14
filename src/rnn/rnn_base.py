import tensorflow as tf
from rnn.text import SeqDataLoader


class RnnModelScratch:
    def __init__(self, vocab_size, num_hiddens):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.trainable_variables = self.create_params()

    def __call__(self, x, state):
        x = tf.one_hot(tf.transpose(x), self.vocab_size)
        x = tf.cast(x, tf.float32)
        return self.rnn(x, state)

    def create_params(self):
        num_inputs = num_outputs = self.vocab_size

        def normal(shape):
            return tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

        # 隐藏层参数
        W_xh = tf.Variable(normal((num_inputs, self.num_hiddens)), dtype=tf.float32)
        W_hh = tf.Variable(normal((self.num_hiddens, self.num_hiddens)), dtype=tf.float32)
        b_h = tf.Variable(tf.zeros(self.num_hiddens), dtype=tf.float32)
        # 输出层参数
        W_hq = tf.Variable(normal((self.num_hiddens, num_outputs)), dtype=tf.float32)
        b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        return params

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size)

    def init_state(self, batch_size):
        return (tf.zeros((batch_size, self.num_hiddens)),)

    def rnn(self, inputs, state):
        # inputs的形状：(时间步数量，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = self.trainable_variables
        h, = state
        outputs = []
        # X的形状：(批量大小，词表大小)
        for x in inputs:
            x = tf.reshape(x, [-1, W_xh.shape[0]])
            h = tf.tanh(tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + b_h)
            y = tf.matmul(h, W_hq) + b_q
            outputs.append(y)
        return tf.concat(outputs, axis=0), (h,)


def main():
    data_iter = SeqDataLoader(batch_size=2, num_steps=5, use_random_iter=False, max_tokens=1000)
    corpus, vocab = data_iter.corpus, data_iter.vocab

    x = tf.reshape(tf.range(10), (2, 5))
    num_hiddens = 512

    net = RnnModelScratch(len(vocab), num_hiddens)
    state = net.begin_state(x.shape[0])
    y, new_state = net(x, state)
    print(y.shape, len(new_state), new_state[0].shape)


if __name__ == "__main__":
    main()
