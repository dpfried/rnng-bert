import sys
from get_oracle import get_tags_tokens_lowercase_morphfeats

if __name__ == "__main__":
    input_treebank_file = sys.argv[1]
    with open(input_treebank_file) as f_in:
        for line in f_in: 
            tags, tokens, lc, morph = get_tags_tokens_lowercase_morphfeats(line) 
            print(' '.join(tokens))
