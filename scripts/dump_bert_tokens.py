import sys
from get_oracle import get_tags_tokens_lowercase_morphfeats

from bert_tokenize import Tokenizer
import argparse
import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_treebank_file")
    parser.add_argument("input_raw_file")
    parser.add_argument("--bert_model_dir", default="models/uncased-english-wwm")

    args = parser.parse_args()

    bert_tokenizer = Tokenizer(args.bert_model_dir)
    with open(args.input_treebank_file) as f_treebank, open(args.input_raw_file) as f_raw:
        for line_idx, line in tqdm.tqdm(enumerate(f_treebank),ncols=80): 
            if not line.rstrip():
                next(f_raw)
                continue
            tags, tokens, lc, morph = get_tags_tokens_lowercase_morphfeats(line) 

            sub_tokens = []
            for token in tokens:
                token = token.replace('-LRB-', '(').replace('-RRB-', ')') 
                this_sub_tokens = bert_tokenizer.tokenizer.basic_tokenizer.tokenize(token)
                sub_tokens.extend(this_sub_tokens)
            raw_line = next(f_raw)
            raw_sub_tokens = bert_tokenizer.tokenizer.basic_tokenizer.tokenize(raw_line)
            assert sub_tokens == raw_sub_tokens, "line {}: {} != {}\n{}".format(
                line_idx, sub_tokens, raw_sub_tokens,
                list(zip(sub_tokens, raw_sub_tokens))
            )
            print(' '.join(sub_tokens))
