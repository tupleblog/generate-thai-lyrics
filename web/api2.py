import sys
sys.path.append("../")

import pickle
import torch
import torch.nn.functional as F

import numpy as np

from pythainlp import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import RNN

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
