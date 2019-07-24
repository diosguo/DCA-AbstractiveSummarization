import tensorflow as tf
from argparse import ArgumentParser
from model import DCA_Model


argparse = ArgumentParser()
argparse.add_argument('--agents_num', default=3, type=int, help="the number of agents")
argparse.add_argument('--encode_dim', default=128, type=int, help='dim of encoder output')
argparse.add_argument('--agent_length', default=400, type=int, help='input length of per agents')
argparse.add_argument('--emb_dim', default=300, type=int, help='dimention of embedding')
argparse.add_argument('--batch_size', default=16, type=int, help='batch size')
argparse.add_argument('--contextual_layers_num', default=2, type=int, help='number of contextual encoder layers')
argparse.add_argument('--vocab_size', default=20000, type=int, help='size of vocabulary')
argparse.add_argument('--drop_keep', default=0.5, type=float)
argparse.add_argument('--attention_units', default=100, type=int)
argparse.add_argument('--learning_rate', default=0.01, type=float)
argparse.add_argument('--encoder_layers_num', default=3, type=int)
argparse.add_argument('--decode_len', default=10, type=int)
argparse.add_argument('--mode',default='train',type=str,help='train/decode')
argparse.add_argument('--len_per_agent',default=300, type=int)
args = argparse.parse_args()


def main():
    # todo load dataset
    # todo load vocab, word2id, id2word
    # do train or decode
    model = DCA_Model(args)


if __name__ == '__main__':
    main()
