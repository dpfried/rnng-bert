from remove_dev_unk import get_tags_tokens_lowercase

from get_oracle_gen import unkify
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments:  train file and tree file to unk')
    dictionary_file = open(sys.argv[1])
    words_list = set(line.strip() for line in dictionary_file)
    dictionary_file.close()

    with open(sys.argv[2]) as f:
        for i, line in enumerate(f):
            if i % 1000 == 0:
                sys.stderr.write("\radd_dev_unk %d" % i)
            assert('|||' not in line)
            parse = line.strip()

            tags, tokens, lowercase = get_tags_tokens_lowercase(parse)
            assert len(tags) == len(tokens)
            assert len(tokens) == len(lowercase)

            unkified = unkify(tokens, words_list)

            for ix, (tag, token, unk) in enumerate(zip(tags, tokens, unkified)):
                to_rep = '(' + tag + ' ' + token + ')'
                assert(to_rep in parse)
                parse = parse.replace(to_rep, '%s ' % unk, 1)
            print(parse)
