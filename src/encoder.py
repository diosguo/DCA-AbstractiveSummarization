from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Bidirectional, CuDNNLSTM, Dropout, Embedding
from tensorflow.python.keras.activations import tanh
import tensorflow as tf



class Encoder(Model):
    def __init__(self, vocab_size, emb_dim, dropout, encode_dim, agents_num, layers_num, batch_size):
        """
        Include Local Encoder and Contextual Encoder
        :param vocab_size: size of vocab
        :param emb_dim: embedding dim
        :param dropout: dropout keep rate
        :param encode_dim: encode dim
        :param agents_num: number of agents
        :param layers_num: number of layers, sum of Local and Contextual

        :return encoder output with shape: [agents, [batch_size, part_len, encode_dim]] outer is a list in python inner
            is a tensor
        """
        super(Encoder, self).__init__()
        self.local_encoder = LocalEncoder(vocab_size,emb_dim, dropout,encode_dim,agents_num)

        self.context_encoder = ContextualEncoder(layers_num, agents_num, encode_dim, emb_dim, batch_size)



    def call(self, inputs):
        """
        :param inputs: source word id with shape [batch_size, sequence_length] which is word id
        :return: encoder output with shape: [agents, [batch_size, part_len, encode_dim]] outer is a list in python inner
            is a tensor
        """
        return self.context_encoder(
            self.local_encoder(
                inputs
            )
        )


class LocalEncoder(Model):
    def __init__(self, vocab_size, emb_dim, dropout, encode_dim, agents_num):
        super(LocalEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.encode_dim = encode_dim
        self.agents_num = agents_num

        # bilstm
        self.bilstm = Bidirectional(CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat')
        self.dropout_layer = Dropout(self.dropout)

        # define linear
        self.dense = Dense(self.encode_dim)

        self.embedding = Embedding(vocab_size, emb_dim)
        self.embedding_dropout = Dropout(self.dropout)

    def call(self, inputs):
        local_encoder_outputs = []
        inputs_embedding = self.embedding(inputs)
        inputs_embedding = self.embedding_dropout(inputs_embedding)

        for agent_index in range(self.agents_num):
            part_sent = tf.slice(inputs_embedding, [0, 300*agent_index, 0], [-1, 300, self.emb_dim])

            local_encoder_outputs.append(
                self.dense(
                    self.dropout_layer(
                        self.bilstm(part_sent)
                    )

                )
            )
        return local_encoder_outputs

class ContextualEncoder(Model):
    def __init__(self, layer_num, agents_num, encode_dim, emb_dim , batch_size):
        super(ContextualEncoder, self).__init__()
        self.layer_num = layer_num
        self.agents_num = agents_num
        self.encode_dim = encode_dim
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        # define
        self._contextual_encoder = [ [] for _ in range(self.layer_num)]

        # W3 and W4 matrix in article

        # self.w3 = tf.Variable(tf.random_normal(shape=[1, self.encode_dim, self.encode_dim]), dtype=tf.float32)
        # self.w4 = tf.Variable(tf.random_normal(shape=[1, self.encode_dim, self.encode_dim]), dtype=tf.float32)

        self.w3 = Dense(self.encode_dim)
        self.w4 = Dense(self.encode_dim)
        # add a linear
        self.dense = Dense(self.encode_dim)

        # add bilstm
        self.bi_lstm = Bidirectional(CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat')

    def call(self, local_encoder_outputs):
        for x in local_encoder_outputs:
            self._contextual_encoder[0].append(x)
        for layer_index in range(self.layer_num-1):
            for agent_index in range(self.agents_num):
                z = tf.add_n(
                    [
                        tf.reshape(tf.slice(x, [-1, -1, 0], [-1, 1, self.encode_dim]),
                                   [self.batch_size, 1, self.encode_dim]) for x in self._contextual_encoder[layer_index]
                    ]
                )

                print(z.shape)
                z = z / (self.agents_num - 1)
                z = tf.tile(z, [1, local_encoder_outputs[0].shape[1],1])
                print(z.shape)
                f = tanh(self.w3(self._contextual_encoder[layer_index][agent_index]) + self.w4(z))
                print('f shape', f.shape)
                self._contextual_encoder[layer_index+1].append(
                    self.dense(
                        self.bi_lstm(f)
                    )
                )
        return self._contextual_encoder[-1]
