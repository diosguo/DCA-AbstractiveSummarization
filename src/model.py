from tensorflow.python.keras.layers import Embedding, CuDNNLSTM, Dropout, Dense, Bidirectional, Flatten, Input, Softmax
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.activations import tanh
from tensorflow.python import keras
from encoder import ContextualEncoder
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from attention import BahdanauAttention
from loss import Seq2SeqLoss



class DCA_Model(object):
    def __init__(self, arg, word2id, id2word):
        self.n_agents = arg.agents_num
        self.emb_dim = arg.emb_dim
        self.encode_dim = arg.encode_dim
        self.batch_size = arg.batch_size
        self.encoder_layers_num = arg.encoder_layers_num
        self.vocab_size = arg.vocab_size
        self.dropout_keep = arg.drop_keep
        self.attention_units = arg.attention_units
        self.decode_len = arg.decode_len
        self.learning_rate = arg.learning_rate
        self.word2id = word2id
        self.id2word = id2word

        self.model = self._build_model()



    def _build_model(self):

        sequence_source_id = Input([900], name='source_id', dtype=tf.int32)
        sequence_target_id = Input([200], name='target_id', dtype=tf.int32)
        sequence_target_mask = Input([200], name='target_mask', dtype=tf.int32)

        self.encoder = Encoder(self.vocab_size,
                               self.emb_dim,
                               self.dropout_keep,
                               self.encode_dim,
                               self.n_agents,
                               self.encoder_layers_num)
        self.decode = Decoder(self.attention_units, self.encode_dim, self.decode_len, self.vocab_size, self.emb_dim)
        self.softmax = Softmax()
        encoder_outputs, encoder_hidden = self.encoder(sequence_source_id)

        decode_hidden = encoder_hidden
        # 初始输入

        decoder_output = self.decode(sequence_target_id, decode_hidden, encoder_outputs)

        vocab_dists = self.softmax(decoder_output)

        self.loss = Seq2SeqLoss(sequence_target_mask, self.batch_size)

        model = Model([sequence_source_id, sequence_target_id], decoder_output)
        model.compile(Adam(self.learning_rate),loss=self.loss)
        return model


    def train(self):
        self.model.fit()


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
    argparse.add_argument('--learning_rate', default=0.01, type=float)
    arg = argparse.parse_args()

    s = DCA_Model(arg)






