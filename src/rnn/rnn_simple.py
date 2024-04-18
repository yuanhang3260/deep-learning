import tensorflow as tf

import base
from rnn.text import SeqDataLoader
from rnn import rnn_base


class RnnModelSimple(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, vocab_size, **kwargs):
        super(RnnModelSimple, self).__init__(**kwargs)
        rnn_cell = tf.keras.layers.SimpleRNNCell(
            units=num_hiddens,
            kernel_initializer='glorot_uniform'
        )
        self.rnn = tf.keras.layers.RNN(
            cell=rnn_cell,
            time_major=True,
            return_sequences=True,
            return_state=True
        )
        self.vocab_size = vocab_size
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, *args, **kwargs):
        state, = args
        x = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        y, *state = self.rnn(x, state)
        output = self.dense(tf.reshape(y, (-1, y.shape[-1])))
        return output, state

    def begin_state(self, batch_size, dtype=None):
        return self.rnn.cell.get_initial_state(batch_size=batch_size, dtype=dtype)


def main():
    # Load text data and vocab.
    batch_size, num_steps = 32, 35
    data_iter = SeqDataLoader(
        batch_size=batch_size,
        num_steps=num_steps,
        use_random_iter=False,
        max_tokens=10000
    )
    corpus, vocab = data_iter.corpus, data_iter.vocab

    # Create model.
    num_hiddens = 512
    model = RnnModelSimple(num_hiddens, len(vocab))
    #print(rnn_base.predict(prefix='time traveller ', num_preds=10, model=model, vocab=vocab))

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
        loss_mean = rnn_base.train_epoch(data_iter, model, loss, optimizer)
        #test_ppl = mnist_base.accuracy(model(test_images), test_labels)
        print(f"epoch {epoch}, train loss {loss_mean}")

        epochs.append(epoch)
        losses.append(loss_mean)
    print('# Training finished.')

    print(rnn_base.predict(prefix='time traveller', num_preds=50, model=model, vocab=vocab))
    print(rnn_base.predict(prefix='traveller', num_preds=50, model=model, vocab=vocab))

    base.plot_metrics(epochs, metrics=[losses], legends=['train loss'])


if __name__ == "__main__":
    main()
