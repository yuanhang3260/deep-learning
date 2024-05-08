import tensorflow as tf
import numpy as np

import attention_base
import rnn.encode


class PositionWiseFFN(tf.keras.layers.Layer):
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        #self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens, activation='relu')
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, x):
        return self.dense2(self.relu(self.dense1(x)))
        #return self.dense2(self.dense1(X))


class AddNorm(tf.keras.layers.Layer):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization(normalized_shape)

    def call(self, X, Y, **kwargs):
        # residual block followed by layer norm
        return self.layer_norm(self.dropout(Y, **kwargs) + X)


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads,
                 norm_shape, ffn_num_hiddens, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = attention_base.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)


class TransformerEncoder(rnn.encode.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, num_heads, norm_shape, ffn_num_hiddens,
                 num_layers, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = attention_base.PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, num_heads, norm_shape,
            ffn_num_hiddens, dropout, bias)
            for _ in range(num_layers)
        ]
        self.attention_weights = None

    def call(self, x, valid_lens, **kwargs):
        # 因为位置编码值在[-1, 1]之间，因此embed值乘以embed维度的平方根进行缩放，然后再与位置编码相加。
        scale = tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32))
        x = self.pos_encoding(self.embedding(x) * scale, **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(x, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads,
                 norm_shape, ffn_num_hiddens, dropout, i, **kwargs):
        super().__init__(**kwargs)
        self.i = i
        self.attention1 = attention_base.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = attention_base.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat([state[2][self.i], X], axis=1)
        state[2][self.i] = key_values

        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens的开头:(batch_size, num_steps), 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1), shape=(-1, num_steps)),
                repeats=batch_size,
                axis=0)
        else:
            dec_valid_lens = None

        # 自注意力
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens, **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state


class TransformerDecoder(rnn.encode.Decoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, num_heads,
                 norm_shape, ffn_num_hiddens, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = attention_base.PositionalEncoding(num_hiddens, dropout)
        self.blks = [DecoderBlock(
            key_size, query_size, value_size, num_hiddens, num_heads,
            norm_shape, ffn_num_hiddens, dropout, i)
            for i in range(num_layers)
        ]
        self.dense = tf.keras.layers.Dense(vocab_size)
        self._attention_weights = None

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def call(self, X, state, **kwargs):
        scale = tf.math.sqrt(tf.cast(self.num_hiddens, dtype=tf.float32))
        X = self.pos_encoding(self.embedding(X) * scale, **kwargs)
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


def main():
    encoder = TransformerEncoder(vocab_size=200,
                                 key_size=24,
                                 query_size=24,
                                 value_size=24,
                                 num_hiddens=24,
                                 num_heads=8,
                                 norm_shape=[1, 2],
                                 ffn_num_hiddens=48,
                                 num_layers=2,
                                 dropout=0.5)
    valid_lens = tf.constant([3, 2])
    x = tf.ones(shape=(2, 10))
    result = encoder(x, valid_lens, training=False)
    print(result)


if __name__ == "__main__":
    main()
