import sys
sys.path.append("../")

import json
import torch
import re
import time
import deepcut
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datatool import LyricCorpus

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

MAX_NUM_WORD = 300

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

device = torch.device("cpu")

CACHE_TOKENZIE_DICT = {}

with open('../thai-song-model.pt', 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage).to(device)
model.eval()

with open('../corpus_lyrics.json', mode='r') as f:
    corpus = json.load(f)
corpus['dictionary_reverse'] = {int(k): v for k, v in corpus['dictionary_reverse'].items()}
ntokens = len(corpus['dictionary'])

@app.route("/")
def index():
    num_word = request.args.get('num_word', 150)
    start_word = request.args.get('start_word', '')

    num_word = MAX_NUM_WORD if int(num_word) > MAX_NUM_WORD else int(num_word)

    hidden = model.init_hidden(1)

    if len(start_word) > 0:
        if start_word in CACHE_TOKENZIE_DICT.keys():
            words = CACHE_TOKENZIE_DICT.get(start_word, [])
        else:
            words = deepcut.tokenize(start_word)

        input_ids = [corpus['dictionary'].get(word, 0) for word in words]
        for input_id in input_ids:
            input = torch.from_numpy(np.array([[input_id]])).to(device)
            output, hidden = model(input, hidden)
    else:
        input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    temperature = 0.8
    with torch.no_grad():  # no tracking history
        lyric = ""
        for i in range(int(num_word)):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            #print(word_idx)
            input.fill_(word_idx)
            word = corpus['dictionary_reverse'].get(int(word_idx), '')

            lyric += word + ('\n' if i % 20 == 19 else ' ')

    lyric = start_word + lyric.replace('UNKNOWN', '')
    lyric = re.sub(' +', '', lyric)
    data = {'lyric': lyric}
    return jsonify(data)

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port=5555, debug=True)
