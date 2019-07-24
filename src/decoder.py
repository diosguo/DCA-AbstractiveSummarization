from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Bidirectional, CuDNNLSTM, Dropout, Embedding, Concatenate, Lambda
import tensorflow as tf
import keras.backend as K
from attention import HiAttention

class Decoder(Model):
    def __init__(self, attention_units, decode_dim, decode_len, vocab_size, emb_dim, batch_size, word2id):
        super(Decoder, self).__init__()
        print('Building Decoder')
        self.decode_len = decode_len
        self.attention = HiAttention(attention_units)
        self.lstm = CuDNNLSTM(decode_dim
                              # return_sequences=True,
                              # recurrent_initializer='glorot_uniform',
                              # return_state=True
                              )
        self.decode_dim = decode_dim
        # 现在用了Decoder Encoder独立的Embedding
        # todo 两个Embedding合并
        self.embedding = Embedding(vocab_size, emb_dim)
        self.batch_size = batch_size
        # decode to vocab
        self.dense = Dense(vocab_size)
        self.word2id = word2id

    def call(self, target_id, encoder_outputs):

        unit_input = K.expand_dims([self.word2id['<start>']]*self.batch_size,1)

        output = []

        # last state of first agent
        hidden = encoder_outputs[0][:,-1,:]

        pre_context_vector = K.variable(K.zeros(shape=[self.batch_size, self.decode_dim]))
        for i in range(self.decode_len):
            print('build %d step of %d' % (i,self.decode_len))
            context_vector = self.attention(hidden, encoder_outputs)

            target_emb = self.embedding(unit_input)

            target_emb = K.concatenate([K.expand_dims(context_vector, 1), target_emb], axis=-1)

            step_output = self.lstm(target_emb)

            hidden = step_output

            # step_output = tf.reshape(step_output, (-1, step_output.shape[1]))  # 0=batch_size, 1=1, 2=decode_dim

            step_output = self.dense(K.concatenate([step_output, context_vector, pre_context_vector],axis=-1))
            pre_context_vector = context_vector

            output.append(step_output)

            unit_input = K.expand_dims(target_id[:, i], 1)

        output = Lambda(lambda x:tf.stack(x, axis=1))(output)
        return output
