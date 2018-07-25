import json
import torch
import re
import random

import flask
from flask import Flask, request


device = torch.device("cpu")
app = Flask(__name__,
            template_folder='templates')
app.secret_key = 'made by tupleblog :)'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# load model and corpus
with open('./thai-song-model.pt', 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage).to(device)
model.eval()

with open('./corpus_lyrics.json', mode='rb') as f:
    corpus = json.load(f)
corpus['dictionary_reverse'] = {int(k): v for k, v in corpus['dictionary_reverse'].items()}
ntokens = len(corpus['dictionary'])


@app.route("/", methods=['GET', 'POST'])
def index():
    torch.manual_seed(random.randint(0, 10000))
    num_word = request.args.get('num_word', 150)
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    temperature = 0.8
    with torch.no_grad():  # no tracking history
        lyric = ""
        for i in range(int(num_word)):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus['dictionary_reverse'].get(int(word_idx), '')
            lyric += word + ('\n' if i % 20 == 19 else ' ')
    lyric = re.sub(' +', '', lyric)
    data = {'lyric': lyric.replace('\n', '</br>'), 'len': len}
    return flask.render_template('index.html', **data)

if __name__ == '__main__':
    app.run(port=5555, debug=True)