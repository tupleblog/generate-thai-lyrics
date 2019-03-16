import sys
sys.path.append("../")

import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from pythainlp import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS

class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, 
                                      embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, 
                            hidden_size=self.hidden_dim, 
                            dropout=self.dropout,
                            num_layers=self.n_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
    
    
    def forward(self, nn_input, hidden):
        batch_size, _ = nn_input.size() # batch first
        embedding_input = self.embedding(nn_input)
        nn_output, hidden = self.lstm(embedding_input, hidden)
        nn_output = nn_output.contiguous().view(-1, self.hidden_dim)
        
        output = self.fc(nn_output)
        output = output.view(batch_size, -1, self.output_size)
        output = output[:, -1]

        # return one batch of output word scores and the hidden state
        return output, hidden
    
    
    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        
        # initialize hidden state with zero weights, and move to GPU if available
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

MAX_NUM_WORD = 300

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

device = torch.device("cpu")
train_on_gpu = torch.cuda.is_available()
train_on_gpu = False

vocab_to_int = pickle.load(open('modelv2/vocab_to_int.pkl', 'rb'))
int_to_vocab = pickle.load(open('modelv2/int_to_vocab.pkl', 'rb'))

trained_rnn = torch.load('modelv2/lstm_model.pt', map_location=lambda storage, loc: storage).to(device)
trained_rnn.eval()

def generate(rnn, start_word, int_to_vocab, pad_value, predict_len=100):
    rnn.eval()
    
    words = word_tokenize(start_word)
    start_word_ids = []
    predicted = words
    
    word_ids = [vocab_to_int.get(word, pad_value) for word in words]
    current_seq = np.array([np.pad(word_ids, (20-len(word_ids), pad_value), 'constant')])
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 100
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq, -1, 1)
        current_seq[-1][-1] = word_i
        #print(current_seq)
    gen_sentences = ''.join(predicted)    
    return gen_sentences

@app.route("/")
def index():

    start_word = request.args.get('start_word', '')
    num_word = request.args.get('num_word', 150)
    num_word = MAX_NUM_WORD if int(num_word) > MAX_NUM_WORD else int(num_word)

    generated_lyric = generate(trained_rnn, start_word, 
                            int_to_vocab, 0, num_word)

    data = {'lyric': generated_lyric, 'v': 2}
    return jsonify(data)

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=5555, debug=True)
