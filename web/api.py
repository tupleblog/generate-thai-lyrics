import sys
sys.path.append("../")

import json
import torch
import re
import time
from flask import Flask, request
from datatool import LyricCorpus

app = Flask(__name__)
app.debug = True

device = torch.device("cpu")

with open('../thai-song-model.pt', 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage).to(device)
model.eval()

with open('../corpus_lyrics.json', mode='r') as f:
    corpus = json.load(f)
corpus['dictionary_reverse'] = {int(k): v for k, v in corpus['dictionary_reverse'].items()}
ntokens = len(corpus['dictionary'])

@app.route("/")
def hello():
    num_word = request.args.get('num_word', 150)
    print("num_word: ", num_word)
    start = time.time()
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    end = time.time()
    print("xxx: " + str(end - start))
    start = time.time()
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

    lyric = lyric.replace('UNKNOWN', '')
    lyric = re.sub(' +', '', lyric)
    end = time.time()
    print(end - start)
    return "<pre>" + lyric + "</pre>"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5555, debug=True)
