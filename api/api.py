import sys

sys.path.append("../")

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import torch
import torch.nn.functional as F

import numpy as np
import deepcut
import math


from pythainlp.tokenize import sent_tokenize, word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import RNN, TransformerModel

import functools
import timeit


MAX_NUM_WORD = 300

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app)

device = torch.device("cpu")
train_on_gpu = torch.cuda.is_available()
train_on_gpu = False

vocab = pickle.load(open("models/transformer/vocab_siamzone-v4-space.pkl", "rb"))

vocab_to_int = vocab["vocab_to_int"]
int_to_vocab = vocab["int_to_vocab"]

ntokens = len(vocab_to_int)
emsize = 512
nhid = 512
nlayers = 4
nhead = 4
dropout = 0.2

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

model_save_path = "./models/transformer/lm-siamzone-v4-space-342.pkl"
model.load_state_dict(torch.load(model_save_path, map_location=torch.device("cpu")))
model.eval()

print("Model initialized")


def top_k_top_p_filtering(logits, top_k, top_p, temperature, filter_value=-float("Inf")):
    # Hugging Face script to apply top k and nucleus sampling
    logits = logits / temperature

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


@functools.lru_cache(maxsize=128)
def tokenize_ids(words):
    print("tokenize_ids: ", words)

    tokens = word_tokenize(words, engine="deepcut")

    word_ids = [vocab_to_int.get(w, 0) for w in tokens]

    return word_ids


def predict(model, start_word="อยากจะไป", size=50):
    # start_ids = torch.LongTensor([vocab_to_int.get(w, 0) for w in deepcut.tokenize(start_word)])

    word_ids = tokenize_ids(start_word)
    start_ids = torch.LongTensor(word_ids)

    print(start_ids)

    outputs = []
    outputs.extend(start_ids.tolist())
    for i in range(size):
        data = torch.tensor(outputs).unsqueeze(-1).to(device)
        src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        logits = model.forward(data, src_mask)
        top_k, top_p, temperature = 100, 0.95, 1
        filtered_logits = top_k_top_p_filtering(
            logits[-1].squeeze(), top_k=top_k, top_p=top_p, temperature=temperature
        )
        probabilities = F.softmax(filtered_logits, dim=-1)
        probabilities_logits, probabilities_position = torch.sort(probabilities, descending=True)
        predicted_token = torch.multinomial(probabilities, 1)
        outputs.append(int(predicted_token))

    return [int_to_vocab.get(idx, " ") for idx in outputs]


@app.route("/")
def index():

    start_time = timeit.default_timer()

    start_word = request.args.get("start_word", "")
    num_word = request.args.get("num_word", 50)
    num_word = MAX_NUM_WORD if int(num_word) > MAX_NUM_WORD else int(num_word)

    generated_lyric = predict(model, start_word, num_word)

    # Add new line
    # sentences = generated_lyric.split(" ")
    sentences = ""
    for w in generated_lyric:
        if w in ["<unk>"]:
            continue
        sentences += w

    sentences = sentences.split(" ")
    lines = []
    current_line = ""
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)

        current_line = current_line + " " + sentence
        if len(current_line) > 20:
            lines.append(current_line.strip())
            lines.append("\n")
            current_line = ""

    usage_time = round(timeit.default_timer() - start_time, 3)

    data = {"lyric": "".join(lines), "usage_time": usage_time, "v": 3}
    return jsonify(data)


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=8585, debug=True)
