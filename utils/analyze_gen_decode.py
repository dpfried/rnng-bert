import sys

from analyze_sampling import acc_tpl

def compare_lists(xs, ys):
    x_larger = 0
    y_larger = 0
    tie = 0
    for x, y in zip(xs, ys):
        if x == y:
            tie += 1
        elif x > y:
            x_larger += 1
        else:
            y_larger += 1
    return x_larger, tie, y_larger

def gold_and_pred_for_file(fname, limit=None):
    gold_scores = []
    pred_scores = []

    print(fname)
    with open(fname) as f:
        for line in f:
            if line.startswith("gold nlp:"):
                gold_scores.append(float(line.strip().split()[-1]))
            elif line.startswith("pred nlp"):
                pred_scores.append(float(line.strip().split()[-1]))

    N_gold = len(gold_scores)
    N_pred = len(pred_scores)
    print("%d gold, %d pred" % (N_gold, N_pred))
    assert(N_gold == N_pred or N_gold == N_pred + 1)  # in case cut off in the middle of pred

    N = min(N_gold, N_pred)
    if limit is not None:
        print("limiting to %d" % limit)
        N = min(limit, N)
    return gold_scores[:N], pred_scores[:N], N

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', nargs='+')
    parser.add_argument('--limit', type=int)

    args = parser.parse_args()

    f1 = args.fnames[0]
    gold_1, pred_1, N1 = gold_and_pred_for_file(f1, limit=args.limit)

    # rev order because negative log probs, higher is worse
    pred_better, tie_1, gold_better = compare_lists(gold_1, pred_1)

    print("%d / %d gold\t(%0.1f%%)" % acc_tpl(gold_better, N1))
    print("%d / %d tie \t(%0.1f%%)" % acc_tpl(tie_1, N1))
    print("%d / %d pred\t(%0.1f%%)" % acc_tpl(pred_better, N1))

    if len(args.fnames) >= 2:
        assert(len(args.fnames) == 2)
        f2 = args.fnames[1]
        print()
        gold_2, pred_2, N2 = gold_and_pred_for_file(f2, limit=args.limit)

        N = min(N1, N2)
        print("comparing first %d sents" % N)

        two_better, tie, one_better = compare_lists(pred_1[:N], pred_2[:N])
        print("%d / %d (%0.1f%%)\t"% acc_tpl(one_better, N) + f1 )
        print("%d / %d (%0.1f%%)\ttie" % acc_tpl(tie, N))
        print("%d / %d (%0.1f%%)\t" % acc_tpl(two_better, N) + f2 )
