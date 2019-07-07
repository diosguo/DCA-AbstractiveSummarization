from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dropout, Dense, Bidirectional, Flatten, Input
from tensorflow.python.keras.models import Sequential
import tensorflow as tf
from tensorflow.python.keras.models import

class DCA_Model(object):
    def __init__(self, arg):
        self.n_agents = arg.agents_num
        self.emb_dim = arg.emb_dim
        self.encode_dim = arg.encode_dim
        self.batch_size = arg.batch_size

        self._build_local_encoder()
        self._build_contextual_encoder()


    def _build_local_encoder(self):
        self._local_encoder = []
        input = Input([900],batch_size=self.batch_size)
        e = Embedding(1, self.emb_dim, input_length=900)
        e = Dropout(0.2)(e(input))
        for i in range(self.n_agents):
            et = tf.slice(e,[0,300*i,0],[self.batch_size, 300, self.emb_dim])


            b_lstm = Bidirectional(CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat')(et)
            b_lstm = Dropout(0.8)(b_lstm)
            self._local_encoder.append(b_lstm)

    def _build_contextual_encoder(self):
        self._contextual_encoder = []
        v = tf.Variable(0,dtype=tf.float32)
        w3 = tf.Variable(0, dtype=tf.float32)
        w4 = tf.Variable(0, dtype=tf.float32)
        for i in range(self.n_agents):
            z = tf.add_n([ tf.slice(x,[-1,-1,0],[1,1,self.encode_dim*2]) for x in self._local_encoder])
            z = z/self.n_agents
            f = v*tf.tanh((w3*self._local_encoder[i]+w4*z))
            self._contextual_encoder.append(Bidirectional(CuDNNLSTM(self.encode_dim,return_sequences=True),
                                                          merge_mode='concat')(f))


if __name__ == '__main__':
    from argparse import ArgumentParser

    argparse = ArgumentParser()
    argparse.add_argument('--agents_num', default=3, type=int, help="the number of agents")
    argparse.add_argument('--encode_dim', default=300, type=int, help='dim of encoder output')
    argparse.add_argument('--agent_length', default=400, type=int, help='input length of per agents')
    argparse.add_argument('--emb_dim', default=300, type=int, help='dimention of embedding')
    argparse.add_argument('--batch_size', default=16, type=int, help='batch size')
    arg = argparse.parse_args()

    s = DCA_Model(arg)






