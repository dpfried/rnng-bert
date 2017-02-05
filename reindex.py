import sys

from remove_dev_unk import get_tags_tokens_lowercase

def parse_line(line):
    return line.strip().split(' ||| ')

if __name__ == "__main__":
    file_with_correct_indices = sys.argv[1]
    file_to_update_indices = sys.argv[2]

    with open(file_with_correct_indices) as f_cor, open(file_to_update_indices) as f_upd:
        for cor, to_upd in zip(f_cor, f_upd):
            cor_toks = parse_line(cor)
            to_upd_toks = parse_line(to_upd)
            cor_tags, cor_words, _ = get_tags_tokens_lowercase(cor_toks[2].strip())
            to_up_tags, to_up_words, _ = get_tags_tokens_lowercase(to_upd_toks[2].strip())
            assert(cor_tags == to_up_tags)
            assert(cor_words == to_up_words)
            # correct index, new score, new parse
            print("%s ||| %s ||| %s" % (cor_toks[0], to_upd_toks[1], cor_toks[2]))

