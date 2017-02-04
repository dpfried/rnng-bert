import sys

def parse_line(line):
    return line.strip().split(' ||| ')

if __name__ == "__main__":
    file_with_correct_indices = sys.argv[1]
    file_to_update_indices = sys.argv[2]

    with open(file_with_correct_indices) as f_cor, open(file_to_update_indices) as f_upd:
        for cor, to_upd in zip(f_cor, f_upd):
            cor_toks = parse_line(cor)
            to_upd_toks = parse_line(to_upd)
            assert(cor_toks[2].strip() == to_upd_toks[2].strip())
            # correct index, new score, new parse
            print("%s ||| %s ||| %s" % (cor_toks[0], to_upd_toks[1], to_upd_toks[2]))

