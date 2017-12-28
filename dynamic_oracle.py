from collections import namedtuple, Counter
import random

Span = namedtuple("Span", ["label", "start", "end"])
State = namedtuple("State", ["position", "last_action", "open_spans"])

MAX_OPEN_NTS = 100
MAX_CONS_NT = None

NTS = ['($', "(''", '(*HASH*', '(,', '(-LRB-', '(-RRB-', '(.', '(:', '(ADJP', '(ADVP', '(CC', '(CD', '(CONJP', '(DT', '(EX', '(FRAG', '(IN', '(INTJ', '(JJ', '(JJR', '(JJS', '(LS', '(LST', '(MD', '(NAC', '(NN', '(NNP', '(NNPS', '(NNS', '(NP', '(NX', '(PDT', '(POS', '(PP', '(PRN', '(PRP', '(PRP$', '(PRT', '(QP', '(RB', '(RBR', '(RBS', '(RP', '(S', '(SBAR', '(SBARQ', '(SINV', '(SQ', '(TO', '(UCP', '(VB', '(VBD', '(VBG', '(VBN', '(VBP', '(VBZ', '(VP', '(WDT', '(WHADJP', '(WHADVP', '(WHNP', '(WHPP', '(WP', '(WP$', '(WRB', '(X', '(``']

class FScore(object):
    # taken from https://github.com/jhcross/span-parser/blob/master/src/phrase_tree.py

    def __init__(self, correct=0, predcount=0, goldcount=0):
        self.correct = correct        # correct brackets
        self.predcount = predcount    # total predicted brackets
        self.goldcount = goldcount    # total gold brackets


    def precision(self):
        if self.predcount > 0:
            return (100.0 * self.correct) / self.predcount
        else:
            return 0.0


    def recall(self):
        if self.goldcount > 0:
            return (100.0 * self.correct) / self.goldcount
        else:
            return 0.0


    def fscore(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) > 0:
            return (2 * precision * recall) / (precision + recall)
        else:
            return 0.0
   

    def __repr__(self):
        precision = self.precision()
        recall    = self.recall()
        fscore = self.fscore()
        return '(P= {:0.2f}, R= {:0.2f}, F= {:0.2f})'.format(
            precision, 
            recall, 
            fscore,
        )


    def __iadd__(self, other):
        self.correct += other.correct
        self.predcount += other.predcount
        self.goldcount += other.goldcount
        return self


    def __add__(self, other):
        return Fmeasure(self.correct + other.correct, 
                        self.predcount + other.predcount, 
                        self.goldcount + other.goldcount)


    def __cmp__(self, other):
        return cmp(self.fscore(), other.fscore())

def spans_and_stack(parse):
    stack = []
    words = []
    spans = []
    open_count = 0
    for token in parse:
        if token.startswith("("):
            stack.append((token[1:], len(words)))
            open_count += 1
        elif token.startswith(")"):
            assert open_count > 0
            label, start = stack.pop()
            assert start != len(words)
            spans.append((label, start, len(words)))
            open_count -= 1
        else:
            words.append(token)
    return spans, stack, words

def oracle_action(gold_parse, partial_parse):
    gold_spans, gold_stack, gold_words = spans_and_stack(gold_parse)
    assert not gold_stack

    part_spans, part_stack, part_words = spans_and_stack(partial_parse)
    part_pos = len(part_words)
    remaining_spans = set(gold_spans) - set(part_spans)

    can_reduce = len(part_stack) > 1 or len(part_words) == len(gold_words)

    # should reduce?
    if can_reduce and part_stack and not partial_parse[-1].startswith("("): 
        top_label, top_start = part_stack[-1]
        # can reduce
        gold_ahead = False
        for (l, s, e) in remaining_spans:
            if l == top_label and s == top_start:
                if e == part_pos:
                    return ")"
                elif e > part_pos:
                    gold_ahead = True

        if not gold_ahead:
            return ")"

    # should open nt?
    part_opened_here = [
        l 
        for l, s in part_stack
        if s == part_pos
    ]
    # print "opened_here", part_opened_here

    golds_to_open = [
        (l, s, e)
        for (l, s, e) in reversed(gold_spans)
        if s == part_pos
    ]
    # print "golds_to_open", golds_to_open
    # print
    part_position = 0
    gold_position = 0
    while part_position < len(part_opened_here) and gold_position < len(golds_to_open):
        if part_opened_here[part_position] == golds_to_open[gold_position][0]:
            gold_position += 1
        part_position += 1

    if gold_position < len(golds_to_open):
        return "(" + golds_to_open[gold_position][0]

    # shift
    if part_pos < len(gold_words):
        return gold_words[part_pos]
    else:
        return None

def compute_f1(gold_spans, predicted_spans):
    # note: this won't return the same score as evalb with COLLINS.prm because of punctuation dropping and label equivalences
    gold_counts = Counter(gold_spans)
    predicted_counts = Counter(predicted_spans)
    correct = 0
    for gs in gold_counts:
        if gs in predicted_counts:
            correct += min(gold_counts[gs], predicted_counts[gs])
    return FScore(correct, sum(predicted_counts.values()), sum(gold_counts.values()))

def available_actions(last_action, words_shifted, sent_length, open_brackets):
    can_reduce = partial_parse

def is_forbidden(action, last_action, words_shifted, sent_length, nopen_parens, ncons_nt):
    is_nt = action.startswith("(")
    is_reduce = action.startswith(")")
    is_shift = not (is_nt or is_reduce)
    assert is_shift or is_reduce or is_nt
    if is_nt and nopen_parens > MAX_OPEN_NTS:
        # print "too many open"
        return True
    if is_nt and MAX_CONS_NT is not None and ncons_nt >= MAX_CONS_NT:
        # print "too many consecutive"
        return True
    if nopen_parens == 0:
        # print "nopen_parens == 0"
        return not is_nt

    # only close top-level parens if all words have been shifted
    if nopen_parens == 1 and words_shifted < sent_length:
        if is_reduce:
            # print "can't close top-level without shifting all words"
            return True

    if is_reduce and last_action.startswith("("):
        # print "can't reduce after NT"
        return True
    if is_nt and words_shifted >= sent_length:
        # print "no words left to cover"
        return True

def test_completion(gold_parse, partial_parse=None, corruption_chance=None, add_shift=False, add_reduce=False, add_nt=False):
    if partial_parse is None:
        partial_parse = []

    gold_spans, gold_stack, gold_words = spans_and_stack(gold)

    words_shifted = 0
    last_action = None
    n_actions = 0
    nopen_parens = 0
    ncons_nt = 0
    if partial_parse is not None:
        for action in partial_parse:
            if action.startswith("("):
                nopen_parens += 1
                ncons_nt += 1
            elif action.startswith(")"):
                nopen_parens -= 1
                ncons_nt = 0
            else:
                words_shifted += 1
                ncons_nt = 0
            n_actions += 1

    def forbidden(action):
        return is_forbidden(action, last_action, words_shifted, len(gold_words), nopen_parens, ncons_nt)

    while True:
        #print "partial_parse", partial_parse 
        action = oracle_action(gold_parse, partial_parse)
        if action is None:
            break
        assert not forbidden(action)
        if corruption_chance is not None and random.random() < corruption_chance:
            possible_actions = []
            if add_shift and words_shifted < len(gold_words):
                ac = gold_words[words_shifted]
                if not forbidden(ac):
                    possible_actions.append(ac)
            if add_nt:
                ac = random.choice(NTS)
                if not forbidden(ac):
                    possible_actions.append(ac)
            if add_reduce:
                ac = ")"
                if not forbidden(ac):
                    possible_actions.append(ac)
            if possible_actions:
                action = random.choice(possible_actions)

        partial_parse.append(action)
        if action.startswith("("):
            nopen_parens += 1
            ncons_nt += 1
        elif action.startswith(")"):
            nopen_parens -= 1
            ncons_nt = 0
        else:
            words_shifted += 1
            ncons_nt = 0
        n_actions += 1
        last_action = action
        if n_actions > 1000:
            print "max actions exceeded"
            break

    return partial_parse

def parse_line(line):
    toks = line.replace(")", " ) ").split()
    spans, stack, words = spans_and_stack(toks)
    assert not stack
    return toks

if __name__ == "__main__":
    gold_parse = "(S (A (B b ) (C c ) ) (D (E e ) (F f ) ) )".split()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus")
    parser.add_argument("--corruption_chance", type=float)
    parser.add_argument("--add_shift", action='store_true')
    parser.add_argument("--add_nt", action='store_true')
    parser.add_argument("--add_reduce", action='store_true')
    args = parser.parse_args()
    if args.corpus:
        with open(args.corpus) as f:
            trees = [parse_line(line) for line in f]
        it = trees
        try:
            import tqdm
            it = tqdm.tqdm(it)
        except:
            pass
        total_fscore = None
        exact_matches = 0
        count = 0
        for gold in it:
            count += 1
            gold_spans, gold_stack, gold_words = spans_and_stack(gold)
            assert not gold_stack

            pred = test_completion(gold, corruption_chance=args.corruption_chance, add_shift=args.add_shift, add_nt=args.add_nt, add_reduce=args.add_reduce)

            pred_spans, pred_stack, pred_words = spans_and_stack(pred)
            assert not pred_stack
            assert gold_words == pred_words

            fscore = compute_f1(gold_spans, pred_spans)
            if total_fscore is None:
                total_fscore = fscore
            else:
                total_fscore += fscore
            if pred == gold:
                exact_matches += 1
            if not args.corruption_chance:
                assert pred == gold
        print str(total_fscore)
        print "exact match: %d / %d (%0.2f%%)" % (exact_matches, count, exact_matches * 100.0 / count)
