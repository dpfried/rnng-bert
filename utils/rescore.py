from analyze_sampling import load_discrim_and_gen
import sys

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n_sentences', type=int)
    parser.add_argument('n_samples', type=int)
    parser.add_argument('discrim_samples_file')
    parser.add_argument('gen_likelihoods_file')
    parser.add_argument('--gen_lambda', type=float, default=1.0)

    args = parser.parse_args()

    sents = load_discrim_and_gen(args.discrim_samples_file, args.gen_likelihoods_file, args.n_samples, include_gold=False, include_gen=False)
    assert(len(sents) == args.n_sentences)

    def scoring_fn(parse):
        return parse.gen_score * args.gen_lambda + parse.discrim_score * (1 - args.gen_lambda)

    last_ix = None
    for sent in sents:
        best = max(sent.samples, key=scoring_fn)
        sys.stdout.write("%s ||| %s ||| %s\n" % (best.index, scoring_fn(best), best.tree))
