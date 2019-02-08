import bert_tokenize
import subprocess
import json
import os
from get_oracle import unkify

bert_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bert_models", "uncased_L-12_H-768_A-12")
bert_tokenizer = bert_tokenize.Tokenizer(bert_model_dir)

dictionary_file = "corpora/english/train.dictionary"
with open(dictionary_file, 'r') as f:
    dictionary = set(line.strip() for line in f)

morph_aware_unking = True

model = "models/inorder_bert_large_bs=32_lr=2e-5_adam_patience=2_best-epoch-14_it-6121-f1-95.71_model"

command = "build/nt-parser/nt-parser \
        --cnn-mem 1000,0,500 \
        --model_dir {}\
        -T corpora/english/in_order/train.oracle \
        --interactive \
        --inorder \
        --bert \
        --bert_large \
        --lstm_input_dim 128 \
        --hidden_dim 128 ".format(model)

def to_json(tokens):
    lc = [token.lower() for token in tokens]
    pos = ["DT" for token in tokens]
    bert_input_ids, bert_word_end_mask = bert_tokenizer.tokenize(tokens)
    unks = unkify(tokens, dictionary, morph_aware_unking)
    data = {
        'raw': ' '.join(tokens),
        'pos': ' '.join(pos),
        'lc': ' '.join(lc),
        'unk': ' '.join(unks),
        'bert_wem': ' '.join(map(str, bert_word_end_mask)),
        'bert_wpi': ' '.join(map(str, bert_input_ids))
    }
    return json.dumps(data)

def parse(proc, tokens):
    try:
        json_data_out = to_json(tokens)
        #print("writing {}".format(json_data_out))
        proc.stdin.write("{}\n".format(json_data_out).encode("utf-8"))
        proc.stdin.flush()
    except:
        print("wrote: {}".format(json_data_out))
        return {'parse': None}
        #print("about to read")
    try:
        line = proc.stdout.readline()
        #print("read {}".format(line))
        json_data_in = json.loads(line)
        return json_data_in['parse']
    except:
        print("read: {}".format(line))

if __name__ == "__main__":
    proc = subprocess.Popen(command.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    while True:
        line = proc.stdout.readline().decode("utf-8")
        #print(line)
        if line.startswith("READY"):
            break

    while True:
        tokens = input("> ").split()
        print(parse(proc, tokens))
