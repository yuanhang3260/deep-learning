import math

import tensorflow as tf

import base
from rnn.text import SeqDataLoader

class RnnModelScratch:
    def __init__(self, vocab_size, num_hiddens):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.trainable_variables = self.create_params()

    def __call__(self, x, state):
        x = tf.one_hot(tf.transpose(x), self.vocab_size)
        x = tf.cast(x, tf.float32)
        return self.rnn(x, state)

    def create_params(self):
        num_inputs = num_outputs = self.vocab_size

        def normal(shape):
            return tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

        # hidden layer
        w_xh = tf.Variable(normal((num_inputs, self.num_hiddens)), dtype=tf.float32)
        w_hh = tf.Variable(normal((self.num_hiddens, self.num_hiddens)), dtype=tf.float32)
        b_h = tf.Variable(tf.zeros(self.num_hiddens), dtype=tf.float32)
        # output layer
        w_hq = tf.Variable(normal((self.num_hiddens, num_outputs)), dtype=tf.float32)
        b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)

        return [w_xh, w_hh, b_h, w_hq, b_q]

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size)

    def init_state(self, batch_size):
        return (tf.zeros((batch_size, self.num_hiddens)),)

    def rnn(self, inputs, state):
        # inputs shape (window_steps, batch_size，vocab_size)
        w_xh, w_hh, b_h, w_hq, b_q = self.trainable_variables
        h, = state
        outputs = []
        # x shape (batch_size，vocab_size)
        for x in inputs:
            x = tf.reshape(x, [-1, w_xh.shape[0]])
            h = tf.tanh(tf.matmul(x, w_xh) + tf.matmul(h, w_hh) + b_h)
            y = tf.matmul(h, w_hq) + b_q
            outputs.append(y)
        return tf.concat(outputs, axis=0), (h,)


def predict(prefix, num_preds, model, vocab):
    state = model.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: tf.reshape(tf.constant([outputs[-1]]), (1, 1)).numpy()
    # Warm up hidden state.
    for y in prefix[1:]:
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    # Predict num_preds steps.
    for _ in range(num_preds):
        y, state = model(get_input(), state)
        outputs.append(tf.argmax(y, axis=1).numpy()[0])
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(grads, theta):
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy() for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad


def train_epoch(train_iter, model, loss, optimizer, use_random_iter=False):
    state = None
    metric = base.Accumulator(size=2)
    for x, y in train_iter:
        # hidden state init
        if state is None or use_random_iter:
            state = model.begin_state(batch_size=x.shape[0], dtype=tf.float32)

        # forward
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = model(x, state)
            y = tf.reshape(tf.transpose(y), (-1))
            train_loss = loss(y, y_hat)

        # backward
        params = model.trainable_variables
        grads = g.gradient(train_loss, params)
        grads = grad_clipping(grads, 1)
        optimizer.apply_gradients(zip(grads, params))

        # metrics
        metric.add(train_loss, 1)

    return math.exp(metric[0] / metric[1])


def main():
    # Load text data and vocab.
    batch_size, num_steps = 32, 35
    data_iter = SeqDataLoader(batch_size=batch_size, num_steps=num_steps, use_random_iter=False, max_tokens=10000)
    corpus, vocab = data_iter.corpus, data_iter.vocab

    # Create model.
    num_hiddens = 512
    model = RnnModelScratch(len(vocab), num_hiddens)

    # x = tf.reshape(tf.range(10), (2, 5))
    # state = model.begin_state(x.shape[0])
    # y, new_state = model(x, state)
    # print(y.shape, len(new_state), new_state[0].shape)

    # print(predict(prefix='time traveller ', num_preds=10, model=model, vocab=vocab))

    # Define loss.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Create optimizer.
    #optimizer = tf.keras.optimizers.SGD(1.0)
    optimizer = tf.keras.optimizers.Adam()

    # Training
    print('# Start training ...')
    epochs, losses = [], []
    num_epochs = 100
    for epoch in range(num_epochs):
        loss_mean = train_epoch(data_iter, model, loss, optimizer)
        #test_ppl = mnist_base.accuracy(model(test_images), test_labels)
        print(f"epoch {epoch}, train loss {loss_mean}")

        epochs.append(epoch)
        losses.append(loss_mean)

    print('# Training finished.')

    print(predict(prefix='time traveller', num_preds=50, model=model, vocab=vocab))
    print(predict(prefix='traveller', num_preds=50, model=model, vocab=vocab))

    base.plot_metrics(epochs, metrics=[losses], legends=['train loss'])


if __name__ == "__main__":
    main()
