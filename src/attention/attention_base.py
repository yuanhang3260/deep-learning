import tensorflow as tf


def sequence_mask(x, valid_len, value=0.):
    # x.shape[0] == valid_len.shape[0]
    maxlen = x.shape[1]
    mask = (tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :]
            < tf.cast(valid_len[:, None], dtype=tf.float32))

    if len(x.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), x, value)
    else:
        return tf.where(mask, x, value)


def masked_softmax(x, valid_lens):
    if valid_lens is None:
        return tf.nn.softmax(x, axis=-1)
    else:
        x_shape = x.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=x_shape[1])
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)

        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        x = sequence_mask(tf.reshape(x, shape=(-1, x_shape[-1])), valid_lens, value=-1e6)
        return tf.nn.softmax(tf.reshape(x, shape=x_shape), axis=-1)


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attention_weights = None

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def call(self, queries, keys, values, valid_lens, **kwargs):
        d = queries.shape[-1]
        scores = (tf.matmul(queries, keys, transpose_b=True) /
                  tf.math.sqrt(tf.cast(d, dtype=tf.float32)))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens, **kwargs):
        # queries，keys，values 形状:
        #   (batch_size，Q/KV的个数，num_hiddens)
        # valid_lens 形状:
        #   (batch_size，)或(batch_size，Q个数)
        #
        # 经过变换后，输出的 queries，keys，values 形状:
        #   (batch_size * num_heads，Q/KV的个数，num_hiddens / num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        # output 形状:
        #   (batch_size * num_heads，Q个数，num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens, **kwargs)

        # output_concat 形状:
        #   (batch_size，Q个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(x, num_heads):
    # 输入X的形状:
    #   (batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:
    #   (batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens / num_heads)
    x = tf.reshape(x, shape=(x.shape[0], x.shape[1], num_heads, -1))

    # 输出X的形状:
    #   (batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens / num_heads)
    x = tf.transpose(x, perm=(0, 2, 1, 3))

    # 最终输出的形状:
    #   (batch_size * num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    return tf.reshape(x, shape=(-1, x.shape[2], x.shape[3]))


def transpose_output(x, num_heads):
    x = tf.reshape(x, shape=(-1, num_heads, x.shape[1], x.shape[2]))
    x = tf.transpose(x, perm=(0, 2, 1, 3))
    return tf.reshape(x, shape=(x.shape[0], x.shape[1], -1))


def main():
    print('---------------------------------------------------------------------')
    #ms = masked_softmax(tf.random.uniform(shape=(3, 2, 5)), tf.constant([2, 4, 3]))
    ms = masked_softmax(tf.random.uniform(shape=(3, 2, 5)), tf.constant([[2, 4], [3, 1], [3, 5]]))
    print(ms)

    print('---------------------------------------------------------------------')
    queries = tf.random.normal(shape=(2, 8, 2))
    keys = tf.ones(shape=(2, 10, 2))
    values = tf.reshape(tf.range(80, dtype=tf.float32), shape=(2, 10, 4))
    valid_lens = tf.constant([2, 6])

    attention = DotProductAttention(dropout=0.5)
    result = attention(queries, keys, values, valid_lens, training=False)
    print(result)

    print('---------------------------------------------------------------------')
    num_hiddens, num_heads = 20, 5
    attention = MultiHeadAttention(key_size=num_hiddens,
                                   query_size=num_hiddens,
                                   value_size=num_hiddens,
                                   num_hiddens=num_hiddens,
                                   num_heads=num_heads,
                                   dropout=0.5)
    batch_size = 2
    num_queries, num_kv = 4, 6
    valid_lens = tf.constant([3, 2])
    x = tf.ones(shape=(batch_size, num_queries, num_hiddens))
    y = tf.ones(shape=(batch_size, num_kv, num_hiddens))
    result = attention(x, y, y, valid_lens, training=False)
    print(result)


if __name__ == "__main__":
    main()
