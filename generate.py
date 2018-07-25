# coding: utf-8
import argparse
import os
import torch
from torch.autograd import Variable

import data
from data import LyricCorpus

parser = argparse.ArgumentParser(description='PyTorch Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device("cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    """
        Loading weights for CPU model while trained on GPU
        https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032/2
    """
    model = torch.load(f, map_location=lambda storage, loc: storage).to(device)
model.eval()

file_name = 'corpus_lyrics.pkl'
if os.path.exists(file_name):
    import pickle
    with open(file_name, mode='rb') as f:
        corpus = pickle.load(f)
else:
    corpus = LyricCorpus(args.data)

ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary_reverse.get(int(word_idx), '')

            outf.write(word + ('\n' if i % 20 == 19 else ''))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
