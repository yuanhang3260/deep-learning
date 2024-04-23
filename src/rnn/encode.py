import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def call(self, x, *args, **kwargs):
        raise NotImplementedError


class Decoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, x, state, *args, **kwargs):
        raise NotImplementedError


class EncoderDecoder(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_input, dec_input, *args, **kwargs):
        enc_outputs = self.encoder(enc_input, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_input, dec_state, **kwargs)
