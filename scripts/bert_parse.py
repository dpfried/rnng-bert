import bert_tokenize
import subprocess
import json
import os
from get_oracle import unkify
import sys

MEMORY_MB = 2000

class Jsonizer(object):
    def __init__(self, model_dir, do_lower_case=True):
        self.model_dir = model_dir
        self.do_lower_case = do_lower_case
        self.bert_tokenizer = bert_tokenize.Tokenizer(model_dir, do_lower_case=do_lower_case)

    def to_json(self, tokens):
        lc = [token.lower() for token in tokens]
        pos = ["XX" for token in tokens]
        bert_input_ids, bert_word_end_mask = self.bert_tokenizer.tokenize(tokens)
        # TODO(dfried): if we use this for some non-BERT models (or generative 
        # rescoring) that use unks, fix this to actually unk words
        unk = lc
        data = {
            'raw': ' '.join(tokens),
            'pos': ' '.join(pos),
            'lc': ' '.join(lc),
            'unk': ' '.join(unk), 
            'bert_wem': ' '.join(map(str, bert_word_end_mask)),
            'bert_wpi': ' '.join(map(str, bert_input_ids))
        }
        return json.dumps(data, ensure_ascii=False)

def get_graph(language, bert_large):
    if language == 'chinese':
        assert not bert_large, "BERT does not currently have a large model for Chinese"
        return "bert_models/chinese_L-12_H-768_A-12_graph.pb"
    elif language == 'english':
        if bert_large:
            return "bert_models/uncased_L-24_H-1024_A-16_graph.pb"
        else:
            return "bert_models/uncased_L-12_H-768_A-12_graph.pb"
    else:
        assert "invalid language {} (must be english or chinese)".format(language)

def make_command(model_dir, bert_graph_path, bert_large=True, inorder=True, beam_size=None):
    command = [
        "build/nt-parser/nt-parser",
        "--cnn-mem", "{},0,500".format(MEMORY_MB),
        "--model_dir", model_dir,
        "--interactive",
        "--text_format",
        "--bert",
        "--lstm_input_dim", "128",
        "--hidden_dim", "128",
        "--bert_graph_path", bert_graph_path,
    ]
    if inorder:
        command.append("--inorder")
    if bert_large:
        command.append("--bert_large")
    if beam_size is not None:
        command += ["--beam_size", str(beam_size)]
    return command


def send_json_data(proc, json_data_out):
    proc.stdin.write("{}\n".format(json_data_out).encode("utf-8"))
    proc.stdin.flush()

def parse(jsonizer, proc, tokens):
    json_data_out = jsonizer.to_json(tokens)
    try:
        send_json_data(proc, json_data_out)
    except Exception as e:
        print("exception, wrote: {}".format(json_data_out), file=sys.stderr)
        print(e, file=sys.stderr)
        return {'parse': None}
    try:
        line = proc.stdout.readline()
        json_data_in = json.loads(line.decode('utf-8'))
        return json_data_in['parse']
    except Exception as e:
        print("exception, read: {}".format(line), file=sys.stderr)
        print(e, file=sys.stderr)

if __name__ == "__main__":
    import argparse
    import fileinput
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("tokenized_files_to_parse", nargs="*")
    parser.add_argument("--language", choices=['english', 'chinese'], default='english')
    parser.add_argument("--beam_size", type=int)

    args = parser.parse_args()

    model_path = args.model_path
    while model_path.endswith("/"):
        model_path = model_path[:-1]

    files_to_parse = args.tokenized_files_to_parse
    if not files_to_parse:
        files_to_parse = ["-"]

    inorder = True
    bert_large = (args.language == 'english')

    bert_graph_path = get_graph(args.language, bert_large)

    jsonizer = Jsonizer(model_path, do_lower_case=True)

    command = make_command(model_path, bert_graph_path=bert_graph_path, bert_large=bert_large, inorder=inorder, beam_size=args.beam_size)

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    while True:
        line = proc.stdout.readline().decode("utf-8")
        if line.startswith("READY"):
            break
    
    for line in fileinput.input(files=files_to_parse):
        if line.strip():
            tokens = line.split()
            print(parse(jsonizer, proc, tokens))

    send_json_data(proc, json.dumps({"action": "exit"}))
