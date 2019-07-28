import tensorflow as tf
from argparse import ArgumentParser
from model import DCA_Model
import os
from data import Vocab
from batcher import Batcher

argparse = ArgumentParser()
argparse.add_argument('--data_path',type=str)
argparse.add_argument('--agents_num', default=3, type=int, help="the number of agents")
argparse.add_argument('--encode_dim', default=128, type=int, help='dim of encoder output')
argparse.add_argument('--encode_len', default=900, type=int)
argparse.add_argument('--decode_len', default=10, type=int)
argparse.add_argument('--agent_length', default=400, type=int, help='input length of per agents')
argparse.add_argument('--emb_dim', default=300, type=int, help='dimention of embedding')
argparse.add_argument('--batch_size', default=16, type=int, help='batch size')
argparse.add_argument('--beam_size', default=4, type=int)
argparse.add_argument('--contextual_layers_num', default=2, type=int, help='number of contextual encoder layers')
argparse.add_argument('--vocab_size', default=20000, type=int, help='size of vocabulary')
argparse.add_argument('--drop_keep', default=0.5, type=float)
argparse.add_argument('--attention_units', default=100, type=int)
argparse.add_argument('--learning_rate', default=0.01, type=float)
argparse.add_argument('--encoder_layers_num', default=3, type=int)
argparse.add_argument('--mode',default='train',type=str,help='train/decode')
argparse.add_argument('--len_per_agent',default=300, type=int)
argparse.add_argument('--log_root',default='./log_root', type=str)
argparse.add_argument('--exp_name',type=str)
argparse.add_argument('--vocab_path', type=str)
argparse.add_argument('--vocab_size', type=str)
argparse.add_argument('--pointer_gen', default=False, type=bool)
args = argparse.parse_args()


def main():

    args.log_root = os.path.join(args.log_root, args.exp_name)
    if not os.path.exists(args.log_root):
        if args.mode == 'train':
            os.mkdir(args.log_root)
        else:
            raise Exception('Logdir do not exist')

    vocab = Vocab(args.vocab_path, args.vocab_size)

    if args.mode == 'decode':
        args.batch_size = args.beam_size

    hparam_list = ['mode', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'pointer_gen']

    hps = {'mode':args.mode,
           'batch_size':args.batch_size,
           'max_dec_steps':args.decode_len,
           'max_enc_steps':args.encode_len,
           'pointer_gen':args.pointer_gen}
    batcher = Batcher(args.data_path, vocab, hps, single_pass=True if args.mode=='decode' else False)


    # todo load dataset
    # todo load vocab, word2id, id2word
    # do train or decode
    model = DCA_Model(args)


if __name__ == '__main__':
    main()
