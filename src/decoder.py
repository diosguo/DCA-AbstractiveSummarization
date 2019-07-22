from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Bidirectional, CuDNNLSTM, Dropout, Embedding
from tensorflow.python.keras.activations import tanh
import tensorflow as tf
from attention import BahdanauAttention

class Decoder(Model):
    def __init__(self, attention_units, decode_dim, decode_len, vocab_size, emb_dim):
        super(Decoder, self).__init__()
        self.decode_len = decode_len
        self.attention = BahdanauAttention(attention_units)
        self.lstm = CuDNNLSTM(decode_dim, return_state=True)

        self.embedding = Embedding(vocab_size, emb_dim)

        # decode to vocab
        self.dense = Dense(vocab_size)

    def call(self, target_id, hidden, encoder_outputs):
        context_vector, attention_weights = self.attention(hidden, encoder_outputs)

        # 现在用了Decoder Encoder独立的Embedding
        # todo 两个Embedding合并

        target_emb = self.embedding(target_id)

        target_emb = tf.concat([tf.expand_dims(context_vector,1), target_emb], axis=1)

        output, state = self.lstm(target_emb)

        output = tf.reshape(output, (-1, output.shape[2]))  # 0=batch_size, 1=1, 2=decode_dim

        output = self.dense(output)

        return output, state, attention_weights








        for i in range(self.decode_len):
            context_vector, attention_weights = attention(encoder_output, encoder_outputs)

            step_decoded, state_h, state_c = self.decoder()




        with tf.variable_scope('decoder'):
            with tf.variable_scope('word_attention'):
                # calc context_vector for each agents

                pass

            with tf.variable_scope('agent_attention'):
                pass

        self._decoder = Bidirectional(
            CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat'
        )(self.decoder_input, initial_state=self._contextual_encoder[0][-1])

        for i in range(self.n_agents):
            pass