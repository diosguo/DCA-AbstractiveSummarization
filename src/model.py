from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dropout, Dense, Bidirectional, Flatten, Input
from tensorflow.python.keras.models import Sequential, Model
import tensorflow as tf

class DCA_Model(object):
    def __init__(self, arg):
        self.n_agents = arg.agents_num
        self.emb_dim = arg.emb_dim
        self.encode_dim = arg.encode_dim
        self.batch_size = arg.batch_size
        self.contextual_layers_num = arg.contextual_layers_num
        self.vocab_size = arg.vocab_size
        self.dropout_keep = arg.drop_keep
        self._build_local_encoder()
        self._build_contextual_encoder()
        self._build_decoder()


    def _build_local_encoder(self):
        with tf.variable_scope('local_encoder'):
            self._local_encoder = []
            self.encoder_input = Input([900],batch_size=self.batch_size)
            self.decoder_input = Input([200],batch_size=self.batch_size)
            e = Embedding(self.vocab_size, self.emb_dim, input_length=900)
            e = Dropout(self.dropout_keep)(e(self.encoder_input))
            for i in range(self.n_agents):
                et = tf.slice(e,[0,300*i,0],[self.batch_size, 300, self.emb_dim])
                b_lstm = Bidirectional(CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat')(et)
                b_lstm = Dropout(self.dropout_keep)(b_lstm)
                b_lstm = Dense(self.encode_dim)(b_lstm)
                self._local_encoder.append(b_lstm)

    def _build_contextual_encoder(self):

        with tf.variable_scope('contextual_encoder'):
            self._contextual_encoder = [[]] * self.contextual_layers_num


            w3 = tf.Variable(tf.random_normal(shape=[self.encode_dim, self.emb_dim]), dtype=tf.float32)
            w4 = tf.Variable(tf.random_normal(shape=[self.encode_dim, self.emb_dim]), dtype=tf.float32)
            for j in range(self.contextual_layers_num):
                for i in range(self.n_agents):
                    z = tf.add_n(
                        [tf.reshape(
                            tf.slice(x, [-1, -1, 0], [1, 1, self.encode_dim]),
                            [1, self.encode_dim]) for x in (self._local_encoder if j==0 else self._contextual_encoder[j])]
                    )
                    z = z/(self.n_agents-1)
                    f = tf.tanh(tf.matmul(self._local_encoder[i], w3)+tf.matmul(z, w4))

                    self._contextual_encoder[j].append(
                        Dense(self.encode_dim)(
                            Bidirectional(
                                CuDNNLSTM(self.encode_dim,return_sequences=True),
                                merge_mode='concat')(f)
                        )
                    )

    def _build_decoder(self):

        with tf.variable_scope('decoder'):
            with tf.variable_scope('word_attention'):
                pass

            with tf.variable_scope('agent_attention'):
                pass

        self._decoder = Bidirectional(
            CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat'
        )(self.decoder_input, initial_state=self._contextual_encoder[0][-1])

        for i in range(self.n_agents):
            pass


from tensorflow.python.keras.models import Sequential
CuDNNLSTM()

if __name__ == '__main__':
    from argparse import ArgumentParser

    argparse = ArgumentParser()
    argparse.add_argument('--agents_num', default=3, type=int, help="the number of agents")
    argparse.add_argument('--encode_dim', default=300, type=int, help='dim of encoder output')
    argparse.add_argument('--agent_length', default=400, type=int, help='input length of per agents')
    argparse.add_argument('--emb_dim', default=300, type=int, help='dimention of embedding')
    argparse.add_argument('--batch_size', default=16, type=int, help='batch size')
    argparse.add_argument('--contextual_layers_num', default=2, type=int, help='number of contextual encoder layers')
    argparse.add_argument('--vocab_size', default=20000, type=int, help='size of vocabulary')
    argparse.add_argument('--drop_keep', default=0.5, type=float)

    arg = argparse.parse_args()

    s = DCA_Model(arg)






