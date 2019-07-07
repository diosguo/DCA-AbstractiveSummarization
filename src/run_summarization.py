import tensorflow as tf
from argparse import ArgumentParser
from model import DCA_Model


argparse = ArgumentParser()
argparse.add_argument('--agents_num', default=3, type=int, help="the number of agents")
argparse.add_argument('--encode_dim', default=300, type=int, help='dim of encoder output')
argparse.add_argument('--agent_length', default=400, type=int, help='input length of per agents')
argparse.add_argument('--emb_dim', default=300, type=int, help='dimention of embedding')
argparse.add_argument('--batch_size', default=16, type=int , help='batch size')
argparse.add_argument('--contextual_layers_num', default=2, type=int, help='number of contextual encoder layers')
arg = argparse.parse_args()



def main():
    model = DCA_Model(arg)


if __name__ == '__main__':
    main()
