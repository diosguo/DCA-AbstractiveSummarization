from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dropout, Dense, Bidirectional, Flatten, Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.activations import tanh
from tensorflow.python import keras
from encoder import ContextualEncoder
from encoder import Encoder
import tensorflow as tf
from attention import BahdanauAttention



class DCA_Model(object):
    def __init__(self, arg):
        self.n_agents = arg.agents_num
        self.emb_dim = arg.emb_dim
        self.encode_dim = arg.encode_dim
        self.batch_size = arg.batch_size
        self.encoder_layers_num = arg.encoder_layers_num
        self.vocab_size = arg.vocab_size
        self.dropout_keep = arg.drop_keep
        self.attention_units = arg.attention_units
        self.decode_len = arg.decode_len

        self._build_contextual_encoder()
        self._build_decoder()

        sequence_source_id = Input([900], name='source_id', dtype=tf.int32)
        sequence_target_id = Input([900], name='target_id', dtype=tf.int32)

        self.encoder = Encoder(self.vocab_size,
                               self.emb_dim,
                               self.dropout_keep,
                               self.encode_dim,
                               self.n_agents,
                               self.encoder_layers_num)


        encoder_outputs = self.encoder(sequence_source_id)













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
    argparse.add_argument('--attention_units', default=100, type=int)
    arg = argparse.parse_args()

    s = DCA_Model(arg)






