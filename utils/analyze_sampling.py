from collections import namedtuple

def acc_tpl(num, denom):
    return (num, denom, num / float(denom) * 100)

Parse = namedtuple("Parse", "index, discrim_score, gen_score, tree, length")
Sentence = namedtuple("Sentence", "gold, samples, gen")

def load_discrim_and_gen(discrim_samples_fname, gen_samples_fname, n_samples, include_gold=False, include_gen=False):
    num_addl_samps = 0
    if include_gold:
        num_addl_samps += 1
    if include_gen:
        num_addl_samps += 1

    sents = []
    parses = []

    counter = 0
    with open(discrim_samples_fname) as discrim_file, open(gen_samples_fname) as gen_file:
        for d_line, g_line in zip(discrim_file, gen_file):
            counter += 1
            length, g_score = g_line.strip().split()
            g_score = -float(g_score) # these are neg llhs
            length = int(length)
            index, d_score, tree = d_line.strip().split(" ||| ")
            tree = tree.strip()
            index = int(index)
            d_score = float(d_score)

            parses.append(Parse(index, d_score, g_score, tree, length))

            if counter == n_samples + num_addl_samps:
                counter = 0
                assert(all([p.index == parses[0].index for p in parses]))
                assert(all([p.length == parses[0].length for p in parses]))
                assert(len(parses) == n_samples + num_addl_samps)
                if include_gen:
                    gen, parses = parses[0], parses[1:]
                else:
                    gen = None
                if include_gold:
                    gold, parses = parses[0], parses[1:]
                else:
                    gold = None
                sents.append(Sentence(gold, parses, gen))
                parses = []
    return sents


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("n_samples", type=int)
    parser.add_argument("--include_gen", action='store_true')
    parser.add_argument("--strict", action='store_true')

    args = parser.parse_args()

    if args.include_gen:
        samples = "expts_sampling/dev_pos_embeddings_s=%d_incl_gold_add_gen.samples" % args.n_samples
        num_addl_samps = 2
    else:
        samples = "expts_sampling/dev_pos_embeddings_s=%d_incl_gold.samples" % args.n_samples
        num_addl_samps = 1

    rescored = samples + ".likelihoods"

    sents = load_discrim_and_gen(samples, rescored, args.n_samples, include_gold=True, include_gen=args.include_gen)

    n_sents = 0
    gold_in_discrim_samps = 0
    gold_best_by_discrim = 0
    gold_best_by_gen = 0

    gen_in_discrim_samps = 0
    gen_best_by_gen = 0
    gen_beats_gold = 0
    gen_beats_gold_and_best = 0

    comp = lambda p, q: p > q if args.strict else p >= q

    for sent in sents:
        n_sents += 1

        # gold
        if any(p.tree == sent.gold.tree for p in sent.samples):
            gold_in_discrim_samps += 1

        if comp(sent.gold.discrim_score, max(p.discrim_score for p in sent.samples)):
            gold_best_by_discrim += 1

        if comp(sent.gold.gen_score, max(p.gen_score for p in sent.samples)):
            gold_best_by_gen += 1

        if args.include_gen:
            # gen
            if any(p.tree == sent.gen.tree for p in sent.samples):
                gen_in_discrim_samps += 1

            if comp(sent.gen.gen_score, max(p.gen_score for p in sent.samples)):
                gen_best = True
                gen_best_by_gen += 1
            else:
                gen_best = False

            if comp(sent.gen.gen_score, sent.gold.gen_score):
                gen_beats_gold += 1
                if gen_best:
                    gen_beats_gold_and_best += 1

    comp_rep = ">" if args.strict else ">="

    print("gold found in discrim samples: %d / %d\t%0.2f%%" % acc_tpl(gold_in_discrim_samps, n_sents))
    print("gold " + comp_rep + " all samps in dscore:   %d / %d\t%0.2f%%" % acc_tpl(gold_best_by_discrim, n_sents))
    print("gold " + comp_rep + " all samps in gscore:   %d / %d\t%0.2f%%" % acc_tpl(gold_best_by_gen, n_sents))

    if args.include_gen:
        print
        print("gen found in discrim samples:  %d / %d\t%0.2f%%" % acc_tpl(gen_in_discrim_samps, n_sents))
        print("gen " + comp_rep + " samps in gscore: %d / %d\t%0.2f%%" % acc_tpl(gen_best_by_gen, n_sents))
        print("gen " + comp_rep + " gold in gscore:  %d / %d\t%0.2f%%" % acc_tpl(gen_beats_gold, n_sents))
        print("gen " + comp_rep + " gold and samps:  %d / %d\t%0.2f%%" % acc_tpl(gen_beats_gold_and_best, n_sents))
