import tensorflow as tf

import base
from rnn import encode
from rnn import seq
from rnn import rnn_base
from metrics import Metrics

class Seq2SeqEncoder(encode.Encoder):
    def __init__(self, vocab_size, embed_dim, hiddens_dim, num_layers, dropout=0.0, **kwargs):
        super().__init__(*kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        rnn_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(hiddens_dim, dropout=dropout) for _ in range(num_layers)])
        self.rnn = tf.keras.layers.RNN(rnn_cell, return_sequences=True, return_state=True)

    def call(self, x, *args, **kwargs):
        x = self.embedding(x)
        output = self.rnn(x, **kwargs)
        state = output[1:]
        return output[0], state


class Seq2SeqDecoder(encode.Decoder):
    def __init__(self, vocab_size, embed_dim, hiddens_dim, num_layers, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        rnn_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.GRUCell(hiddens_dim, dropout=dropout) for _ in range(num_layers)])
        self.rnn = tf.keras.layers.RNN(rnn_cell, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def call(self, x, state, **kwargs):
        x = self.embedding(x)
        context = tf.repeat(tf.expand_dims(state[-1], axis=1), repeats=x.shape[1], axis=1)
        x_and_context = tf.concat((x, context), axis=2)
        rnn_output = self.rnn(x_and_context, state, **kwargs)
        output = self.dense(rnn_output[0])
        return output, rnn_output[1:]


class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len
        self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )

    def sequence_mask(self, x, valid_len, value=0):
        maxlen = x.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
               None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(x.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), x, value)
        else:
            return tf.where(mask, x, value)

    # valid_len: (batch_size,)
    # label: (batch_size,num_steps)
    # pred: (batch_size,num_steps,vocab_size)
    def call(self, label, pred):
        mask = tf.ones_like(label, dtype=tf.float32)
        mask = self.sequence_mask(mask, self.valid_len)
        #label = tf.one_hot(label, depth=pred.shape[-1])
        loss = self.loss_func(label, pred)
        loss = tf.reduce_mean((loss * mask), axis=1)
        return loss


def train_epoch(train_iter, model, optimizer, tgt_vocab):
    metric = base.Accumulator(size=3)
    for x, x_valid_len, y, y_valid_len in train_iter:
        bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * y.shape[0]), shape=(-1, 1))
        dec_input = tf.concat([bos, y[:, :-1]], axis=1)

        # forward
        with tf.GradientTape(persistent=True) as tape:
            y_hat, state = model(x, dec_input, x_valid_len, training=True)
            train_loss = MaskedSoftmaxCELoss(y_valid_len)(y, y_hat)

        # backward
        params = model.trainable_variables
        grads = tape.gradient(train_loss, params)
        grads = rnn_base.grad_clipping(grads, theta=1)
        optimizer.apply_gradients(zip(grads, params))

        # metrics
        num_tokens = tf.reduce_sum(y_valid_len).numpy()
        metric.add(tf.reduce_sum(train_loss), num_tokens, 1)

    return metric[0] / metric[1]


def predict(net, src_sentence, src_vocab, tgt_vocab, num_steps, save_attention_weights=False):
    """序列到序列模型的预测"""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = tf.constant([len(src_tokens)])
    src_tokens = seq.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_x = tf.expand_dims(src_tokens, axis=0)
    enc_outputs = net.encoder(enc_x, enc_valid_len, training=False)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_x = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        y, dec_state = net.decoder(dec_x, dec_state, training=False)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_x = tf.argmax(y, axis=2)
        pred = tf.squeeze(dec_x, axis=0)
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred.numpy())
    splitter = ('' if seq.lang == 'chz' else ' ')
    return (splitter.join(tgt_vocab.to_tokens(tf.reshape(output_seq, shape=-1).numpy().tolist())),
            attention_weight_seq)


def main():
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10

    # Load data.
    data_iter, src_vocab, tgt_vocab = seq.load_data_nmt(batch_size, num_steps, num_examples=640)

    # Define model
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    model = encode.EncoderDecoder(encoder, decoder)

    # Create optimizer.
    # optimizer = tf.keras.optimizers.SGD(1.0)
    optimizer = tf.keras.optimizers.Adam()

    # Training
    print('# Start training ...')
    metrics = Metrics(x_label='epoch', y_label_list=['train_loss'])
    for epoch in range(300):
        loss_mean = train_epoch(data_iter, model, optimizer, tgt_vocab)
        print(f"epoch {epoch}, train_loss {loss_mean}")
        metrics.add(x_value=epoch, y_value_list=[loss_mean])

    print('# Training finished.')

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    chns = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, chn in zip(engs, chns):
        translation, attention_weight_seq = predict(model, eng, src_vocab, tgt_vocab, num_steps)
        print(f'{eng} => {translation}')

    metrics.plot()


if __name__ == "__main__":
    main()
