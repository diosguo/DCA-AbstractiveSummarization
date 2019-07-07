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

        self._build_local_encoder()
        self._build_contextual_encoder()
        self._build_decoder()


    def _build_local_encoder(self):
        self._local_encoder = []
        self.encoder_input = Input([900],batch_size=self.batch_size)
        self.decoder_input = Input([200],batch_size=self.batch_size)
        e = Embedding(1, self.emb_dim, input_length=900)
        e = Dropout(0.2)(e(self.encoder_input))
        for i in range(self.n_agents):
            et = tf.slice(e,[0,300*i,0],[self.batch_size, 300, self.emb_dim])
            b_lstm = Bidirectional(CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat')(et)
            b_lstm = Dropout(0.8)(b_lstm)
            self._local_encoder.append(b_lstm)

    def _build_contextual_encoder(self):
        self._contextual_encoder = [[]] * self.contextual_layers_num
        v = tf.Variable(0,dtype=tf.float32)
        w3 = tf.Variable(0, dtype=tf.float32)
        w4 = tf.Variable(0, dtype=tf.float32)
        for j in range(self.contextual_layers_num):
            for i in range(self.n_agents):
                z = tf.add_n(
                    [ tf.slice(x,[-1,-1,0],[1,1,self.encode_dim*2]) for x in (self._local_encoder if j==0 else self._contextual_encoder[j])]
                )
                z = z/self.n_agents
                f = v*tf.tanh((w3*self._local_encoder[i]+w4*z))
                self._contextual_encoder[j].append(Bidirectional(CuDNNLSTM(self.encode_dim,return_sequences=True),
                                                              merge_mode='concat')(f))

    def _build_decoder(self):
        self._decoder = Bidirectional(
            CuDNNLSTM(self.encode_dim, return_sequences=True), merge_mode='concat'
        )(self.decoder_input, initial_state=self._contextual_encoder[0][-1])

        for i in range(self.n_agents):
            pass




if __name__ == '__main__':
    from argparse import ArgumentParser

    argparse = ArgumentParser()
    argparse.add_argument('--agents_num', default=3, type=int, help="the number of agents")
    argparse.add_argument('--encode_dim', default=300, type=int, help='dim of encoder output')
    argparse.add_argument('--agent_length', default=400, type=int, help='input length of per agents')
    argparse.add_argument('--emb_dim', default=300, type=int, help='dimention of embedding')
    argparse.add_argument('--batch_size', default=16, type=int, help='batch size')
    argparse.add_argument('--contextual_layers_num', default=2, type=int, help='number of contextual encoder layers')
    arg = argparse.parse_args()

    s = DCA_Model(arg)






