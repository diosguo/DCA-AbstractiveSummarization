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

        # 现在用了Decoder Encoder独立的Embedding
        # todo 两个Embedding合并
        self.embedding = Embedding(vocab_size, emb_dim)

        # decode to vocab
        self.dense = Dense(vocab_size)

    def call(self, target_id, hidden, encoder_outputs,word2id):
        batch_size = tf.shape(target_id)[0] # get batch size

        unit_input = tf.expand_dims([word2id['<start>']*batch_size])

        output = []

        for i in range(self.decode_len):
            context_vector, attention_weights = self.attention(hidden, encoder_outputs)

            target_emb = self.embedding(unit_input)
            target_emb = tf.concat([tf.expand_dims(context_vector, 1), target_emb], axis=1)

            step_output, hidden = self.lstm(target_emb)

            step_output = tf.reshape(step_output, (-1, step_output.shape[2]))  # 0=batch_size, 1=1, 2=decode_dim

            step_output = self.dense(step_output)

            output.append(step_output)

            unit_input = tf.expand_dims(target_id[:, i], 1)

        return output
