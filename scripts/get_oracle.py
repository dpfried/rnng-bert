from __future__ import print_function
import sys
import get_dictionary
from get_dictionary import is_next_open_bracket, get_between_brackets
import types
import bert_tokenize
import os
import fileinput

import unicodedata

PAREN_NORM = {
    '(': '-LRB-',
    ')': '-RRB-',
}

PAREN_NORM_LC = {
    key: val.lower()
    for key, val in PAREN_NORM.items()
}

def norm_parens(token, lc=False):
    lookup = PAREN_NORM_LC if lc else PAREN_NORM
    for key, value in lookup.items():
        token = token.replace(key, value)
    return token

def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def reverse_tree(tree):
    import nltk
    if isinstance(tree, nltk.Tree):
        children = []
        for child in tree:
            children.append(reverse_tree(child))
        return nltk.Tree(tree.label(), list(reversed(children)))
    else:
        return tree

# tokens is a list of tokens, so no need to split it again
def unkify(tokens, words_dict, morph_aware=True):
    final = []
    for token in tokens:
        # only process the train singletons and unknown words
        if len(token.rstrip()) == 0:
            final.append('UNK')
        elif not(token.rstrip() in words_dict):
            if not morph_aware:
                final.append('UNK')
                continue;
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            result = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    result = result + '-INITC'
                    if lower in words_dict:
                        result = result + '-KNOWNLC'
                else:
                    result = result + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                result = result + '-CAPS'
            elif hasLower:
                result = result + '-LC'
            if hasDigit:
                result = result + '-NUM'
            if hasDash:
                result = result + '-DASH'
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    result = result + '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                if lower[-2:] == 'ed':
                    result = result + '-ed'
                elif lower[-3:] == 'ing':
                    result = result + '-ing'
                elif lower[-3:] == 'ion':
                    result = result + '-ion'
                elif lower[-2:] == 'er':
                    result = result + '-er'
                elif lower[-3:] == 'est':
                    result = result + '-est'
                elif lower[-2:] == 'ly':
                    result = result + '-ly'
                elif lower[-3:] == 'ity':
                    result = result + '-ity'
                elif lower[-1] == 'y':
                    result = result + '-y'
                elif lower[-2:] == 'al':
                    result = result + '-al'
            final.append(result)
        else:
            final.append(token.rstrip())
    return final

# start_idx = open bracket
#def skip_terminals(line, start_idx):
#    line_end_idx = len(line) - 1
#    for i in range(start_idx + 1, line_end_idx):
#        if line[i] == ')':
#            assert line[i + 1] == ' '
#            return (i + 2)
#    raise IndexError('No close bracket found in a terminal')

def get_tag_token_pairs(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    return output

def get_tags_tokens_lowercase_morphfeats(line, try_to_fix_parens=False):
    output = get_tag_token_pairs(line)
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    output_morphfeats = []
    for terminal in output:
        splits = terminal.split()
        if try_to_fix_parens and not splits:
            splits = ('', '(')
        tag_and_feats = splits[0]
        token_splits = splits[1:]
        if try_to_fix_parens and not token_splits:
            token_splits = [')']
        if len(token_splits) > 1:
            sys.stderr.write("warning: whitespace found in token {} on line {}\n".format(' '.join(token_splits), line.rstrip()))
        #tag_and_feats, token = terminal.split()
        token = ' '.join(token_splits)

        if '##' in tag_and_feats:
            tag, morph_feats, rest = tag_and_feats.split('##')
            assert not rest
            output_morphfeats.append(morph_feats)
        else:
            tag = tag_and_feats
        output_tags.append(tag)
        output_tokens.append(token)
        output_lowercase.append(token.lower())
    return [output_tags, output_tokens, output_lowercase, output_morphfeats]

def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)

def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)

    last_nt = None
    open_nts = 0
    cons_nts = 0
    same_nts = 0
    max_open_nts = 0
    max_cons_nts = 0
    max_same_cons_nts = 0
    # this works because ADV don't contain other NTs, at least for Brown
    in_adv = False

    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if line_strip[i:i+5] == '(ADV ' or line_strip[i:i+5] == '(AUX ':
                in_adv = True
                i += 1
                while line_strip[i] != '(':
                    i += 1
            elif is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                if '-' in curr_NT:
                    curr_NT = curr_NT.split('-')[0]
                output_actions.append('NT(' + curr_NT + ')')

                if curr_NT == last_nt:
                    same_nts += 1
                else:
                    same_nts = 1
                last_nt = curr_NT

                open_nts += 1
                cons_nts += 1

                max_open_nts = max(max_open_nts, open_nts)
                max_cons_nts = max(max_cons_nts, cons_nts)
                max_same_cons_nts = max(max_same_cons_nts, same_nts)

                i += 1
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else:
                # it's a terminal symbol
                cons_nts = 0
                same_nts = 0
                last_nt = None

                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             if not in_adv:
                open_nts -= 1
                cons_nts = 0
                same_nts = 0
                last_nt = None
                output_actions.append('REDUCE')
             in_adv = False
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx
    return output_actions, max_open_nts, max_cons_nts, max_same_cons_nts

# modified from https://github.com/LeonCrashCode/InOrderParser
def construct(actions, trees):
    while len(actions) > 0:
        act = actions[0]
        actions = actions[1:]
        if act[0] == 'N':
            tree = [act]
            actions, tree = construct(actions,tree)
            trees.append(tree)
        elif act[0] == 'S':
            trees.append(act)
        elif act[0] == 'R':
            break;
        else:
            assert False
    return actions, trees

def get_in_order_actions(trees, actions):
    if isinstance(trees[1], list):
        actions = get_in_order_actions(trees[1], actions)
    else:
        actions.append(trees[1])

    assert isinstance(trees[0], str)
    actions.append("NT"+trees[0][2:])
    
    for item in trees[2:]:
        if isinstance(item, list):
            actions = get_in_order_actions(item, actions)
        else:
            actions.append(item)
    actions.append("REDUCE")
    return actions      

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dictionary_file")
    parser.add_argument("corpus_file")
    parser.add_argument("--in_order", action='store_true')
    parser.add_argument("--no_morph_aware_unking", action='store_true')
    parser.add_argument("--bert_model_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uncased_L-12_H-768_A-12"))
    parser.add_argument("--collapse_unary", action='store_true', help='collapse unary chains, with nonterminals separated by "+"')
    parser.add_argument("--reverse_trees", action='store_true', help='treat trees as horizontally mirrored for the sake of traversal orders')
    parser.add_argument("--is_candidate_file", action='store_true')
    parser.add_argument("--is_token_file", action='store_true')
    parser.add_argument("--block_size", type=int)
    parser.add_argument("--only_these_blocks", type=int, nargs='*', default=[])
    parser.add_argument("--base_fname")
    parser.add_argument("--max_bert_len", type=int, default=512, help="chop off tokens at the end of sentences until the sentence has no more than this many wordpieces (as tokenized by BERT)")
    args = parser.parse_args()

    print(' '.join(sys.argv), file=sys.stderr)
    # train_file = open(sys.argv[1], 'r')
    # words_list = set(get_dictionary.get_dict(train_file))
    # train_file.close()
    with open(args.dictionary_file, 'r') as f:
        words_list = set(line.strip() for line in f)

    # dev_lines = dev_file.readlines()
    line_ctr = 0
    max_open_nts = 0
    max_open_nts_ix = None
    max_cons_nts = 0
    max_cons_nts_ix = None
    max_same_cons_nts = 0
    max_same_cons_nts_ix = None
    max_unary_count = 0
    max_unary_count_ix = None
    # get the oracle for the train file
    print("loading BERT tokenizer from %s" % (args.bert_model_dir), file=sys.stderr)
    bert_tokenizer = bert_tokenize.Tokenizer(args.bert_model_dir)
    # use fileinput so that we can pass '-' to read from stdin
    fout = sys.stdout
    for line_ctr, line in enumerate(fileinput.input(files=[args.corpus_file])):
        #line = line.replace('\ufeff', '')
        #line = line.encode('utf-8').decode('utf-8', 'ignore')
        if line_ctr % 1000 == 0:
            sys.stderr.write("\rget oracle %d" % line_ctr)
        if args.block_size is not None:
            block = line_ctr // args.block_size
            if args.only_these_blocks and block not in args.only_these_blocks:
                continue
        line = _clean_text(line)
        if args.base_fname:
            if args.block_size is not None and line_ctr % args.block_size == 0:
                if fout != sys.stdout:
                    fout.close()
                fout = open(args.base_fname + "-block-{}".format(line_ctr // args.block_size), 'w')
            else:
                if fout == sys.stdout:
                    fout = open(args.base_fname, 'w')
        if args.is_candidate_file:
            line = line.split("|||")[-1].strip()
            lines = [line]
        else:
            line = line.strip()
            lines = [l.strip() for l in line.split('|||')]
        for segment in lines:
            if not segment:
                continue
            if args.is_token_file:
                tokens = segment.split()
                tags = ["DT"] * len(tokens)
                lowercase = [tok.lower() for tok in tokens]
                morphfeats = []
            else:
                # assert that the parenthesis are balanced
                if segment.count('(') != segment.count(')'):
                    raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr))
                tags, tokens, lowercase, morphfeats = get_tags_tokens_lowercase_morphfeats(segment)
            assert len(tags) == len(tokens)
            assert len(tokens) == len(lowercase)

            # tokens = [PAREN_NORM.get(token, token) for token in tokens]
            # lowercase = [PAREN_NORM_LC.get(lc, lc) for lc in lowercase]
            tokens = [norm_parens(token, lc=False) for token in tokens]
            lowercase = [norm_parens(lc, lc=True) for lc in lowercase]

            bert_input_ids, bert_word_end_mask, filtered_tokens = bert_tokenizer.tokenize(tokens, return_filtered_sentence=True)

            if len(filtered_tokens) < len(tokens):
                print("WARNING: sentence {} has PTB tokens with no wordpieces: {}".format(line_ctr, tokens[:10]))
                if args.is_token_file:
                    tokens = filtered_tokens
                    tags = ["DT"] * len(tokens)
                    lowercase = [tok.lower() for tok in tokens]
                    morphfeats = []

            if len(bert_input_ids) == 2:
                print("WARNING: removing sentence {} that has no BERT tokens: {}".format(line_ctr, segment.encode()))
                continue

            new_toks = tokens
            while len(bert_input_ids) > args.max_bert_len:
                new_toks = new_toks[:-1]
                bert_input_ids, bert_word_end_mask = bert_tokenizer.tokenize(new_toks)

            if len(new_toks) < len(tokens):
                print("WARNING: truncating sentence {} ({} ...) of length {} to length {}".format(line_ctr, tokens[:10], len(tokens), len(new_toks)))
                tags = tags[:len(new_toks)]
                tokens = new_toks
                lowercase = lowercase[:len(new_toks)]
                morphfeats = morphfeats[:len(new_toks)]

            if args.is_token_file:
                # for compatibility with oracle reader, wrap in brackets
                print('!# ( ' + segment.rstrip() + ' )', file=fout)
            else:
                # first line: the bracketed tree itself itself
                print('!# ' + segment.rstrip(), file=fout)

            print(' '.join(tags), file=fout)
            print(' '.join(tokens), file=fout)
            print(' '.join(lowercase), file=fout)
            unkified = unkify(tokens, words_list, not args.no_morph_aware_unking)
            print(' '.join(unkified), file=fout)
            # print morph features, or an empty line
            print(' '.join(morphfeats), file=fout)

            print(' '.join(map(str, bert_word_end_mask)), file=fout)
            print(' '.join(map(str, bert_input_ids)), file=fout)

            if not args.is_token_file:
                tree_string = segment
                if args.collapse_unary or args.reverse_trees:
                    from nltk import Tree
                    tree = Tree.fromstring(tree_string.rstrip())
                    if args.collapse_unary:
                        tree.collapse_unary(collapseRoot=True, joinChar="+")
                    if args.reverse_trees:
                        tree = reverse_tree(tree)
                    tree_string = tree._pformat_flat(nodesep='', parens='()', quotes=False)

                output_actions, mon, mcn, mscn = get_actions(tree_string)

                if mon > max_open_nts:
                    max_open_nts = mon
                    max_open_nts_ix = line_ctr
                if mcn > max_cons_nts:
                    max_cons_nts = mcn
                    max_cons_nts_ix = line_ctr
                if mscn > max_same_cons_nts:
                    max_same_cons_nts = mscn
                    max_same_cons_nts_ix = line_ctr
                if args.in_order:
                    _, trees = construct(output_actions, [])
                    output_actions = get_in_order_actions(trees[0], [])
                unary_count = 0
                for action_ix, action in enumerate(output_actions):
                    print(action, file=fout)
                    if args.in_order:
                        if action.startswith("SHIFT"):
                            unary_count = 0
                        elif action.startswith("REDUCE"):
                            if action_ix > 0:
                                if output_actions[action_ix-1].startswith("NT"):
                                    unary_count += 1
                                    if unary_count > max_unary_count:
                                        max_unary_count = unary_count 
                                        max_unary_count_ix = line_ctr
                                elif output_actions[action_ix-1].startswith("REDUCE"):
                                    unary_count = 0
                if args.in_order:
                    print('TERM', file=fout)
            print('', file=fout)
    print("", file=sys.stderr)
    if not args.is_token_file:
        print("max open nts: %d, line %d" % (max_open_nts, max_open_nts_ix), file=sys.stderr)
        print("max cons nts: %d, line %d" % (max_cons_nts, max_cons_nts_ix), file=sys.stderr)
        print("max same cons nts: %d, line %d" % (max_same_cons_nts, max_same_cons_nts_ix), file=sys.stderr)
        if args.in_order:
            print("max unary count: %d, line %d" % (max_unary_count, max_unary_count_ix), file=sys.stderr)

if __name__ == "__main__":
    main()
