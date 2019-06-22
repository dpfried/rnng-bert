import bert_tokenize
import subprocess
import json
import os
from get_oracle import unkify
import sys

bert_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bert_models")
bert_tokenizer = bert_tokenize.Tokenizer(bert_model_dir)

dictionary_file = "corpora/english/train.dictionary"
with open(dictionary_file, 'r') as f:
    dictionary = set(line.strip() for line in f)

morph_aware_unking = True

command = "build/nt-parser/nt-parser \
        --cnn-mem 1000,0,500 \
        --model_dir {}\
        --interactive \
	--text_format \
        --inorder \
        --bert \
        --bert_large \
        --lstm_input_dim 128 \
        --hidden_dim 128 "

def to_json(tokens):
    lc = [token.lower() for token in tokens]
    pos = ["XX" for token in tokens]
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

def send_json_data(proc, json_data_out):
    #print("writing {}".format(json_data_out), file=sys.stderr)
    proc.stdin.write("{}\n".format(json_data_out).encode("utf-8"))
    proc.stdin.flush()

def parse(proc, tokens):
    try:
        send_json_data(proc, to_json(tokens))
    except Exception as e:
        print("exception, wrote: {}".format(json_data_out), file=sys.stderr)
        print(e, file=sys.stderr)
        return {'parse': None}
        #print("about to read")
    try:
        line = proc.stdout.readline()
        json_data_in = json.loads(line.decode('utf-8'))
        #print("read {}".format(line), file=sys.stderr)
        return json_data_in['parse']
    except Exception as e:
        print("exception, read: {}".format(line), file=sys.stderr)
        print(e, file=sys.stderr)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    while model_path.endswith("/"):
        model_path = model_path[:-1]

    files_to_parse = sys.argv[2:]
    if not files_to_parse:
        files_to_parse = ["-"]
    import fileinput

    proc = subprocess.Popen(command.format(model_path).split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    while True:
        line = proc.stdout.readline().decode("utf-8")
        #print(line)
        if line.startswith("READY"):
            break
    
    for line in fileinput.input(files=files_to_parse):
        if line.strip():
            tokens = line.split()
            print(parse(proc, tokens))

    send_json_data(proc, json.dumps({"action": "exit"}))
