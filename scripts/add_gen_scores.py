import sys

def parse_line(line):
    return line.strip().split(' ||| ')

if __name__ == "__main__":
    candidate_file = sys.argv[1]
    score_file = sys.argv[2]

    with open(candidate_file) as f:
        candidate_lines = f.readlines()

    with open(score_file) as f:
        score_lines = f.readlines()

    assert(len(candidate_lines) == len(score_lines))

    for candidate_line, score_line in zip(candidate_lines, score_lines):
        sent_length, negative_lp = score_line.strip().split()
        negative_lp = float(negative_lp)
        candidate_toks = parse_line(candidate_line)
        print("%s ||| %s ||| %s" % (candidate_toks[0], -negative_lp, candidate_toks[2]))
