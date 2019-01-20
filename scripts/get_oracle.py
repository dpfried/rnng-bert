from __future__ import print_function
import sys
import get_dictionary
from get_dictionary import is_next_open_bracket, get_between_brackets
import types
import bert_tokenize

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

def get_tags_tokens_lowercase_morphfeats(line):
    output = get_tag_token_pairs(line)
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    output_morphfeats = []
    for terminal in output:
        tag_and_feats, token = terminal.split()
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

    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
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
            else: # it's a terminal symbol
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
             open_nts -= 1
             cons_nts = 0
             same_nts = 0
             last_nt = None
             output_actions.append('REDUCE')
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
    args = parser.parse_args()
    # train_file = open(sys.argv[1], 'r')
    # words_list = set(get_dictionary.get_dict(train_file))
    # train_file.close()
    dictionary_file = open(args.dictionary_file, 'r')
    words_list = set(line.strip() for line in dictionary_file)
    dictionary_file.close()

    dev_file = open(args.corpus_file, 'r')
    # dev_lines = dev_file.readlines()
    line_ctr = 0
    max_open_nts = 0
    max_open_nts_ix = None
    max_cons_nts = 0
    max_cons_nts_ix = None
    max_same_cons_nts = 0
    max_same_cons_nts_ix = None
    # get the oracle for the train file
    for line in dev_file:
        line_ctr += 1
        if line_ctr % 1000 == 0:
            sys.stderr.write("\rget oracle %d" % line_ctr)
        # assert that the parenthesis are balanced
        if line.count('(') != line.count(')'):
            raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr))
        # first line: the bracketed tree itself itself
        print('!# ' + line.rstrip())
        tags, tokens, lowercase, morphfeats = get_tags_tokens_lowercase_morphfeats(line)
        assert len(tags) == len(tokens)
        assert len(tokens) == len(lowercase)
        print(' '.join(tags))
        print(' '.join(tokens))
        print(' '.join(lowercase))
        unkified = unkify(tokens, words_list, not args.no_morph_aware_unking)
        print(' '.join(unkified))
        # print morph features, or an empty line
        print(' '.join(morphfeats))

        bert_input_ids, bert_word_end_mask = bert_tokenize.tokenize(tokens)
        print(' '.join(map(str, bert_word_end_mask)))
        print(' '.join(map(str, bert_input_ids)))

        output_actions, mon, mcn, mscn = get_actions(line)

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
        for action in output_actions:
            print(action)
        if args.in_order:
            print('TERM')
        print('')
    print("max open nts: %d, line %d" % (max_open_nts, max_open_nts_ix), file=sys.stderr)
    print("max cons nts: %d, line %d" % (max_cons_nts, max_cons_nts_ix), file=sys.stderr)
    print("max same cons nts: %d, line %d" % (max_same_cons_nts, max_same_cons_nts_ix), file=sys.stderr)
    dev_file.close()

if __name__ == "__main__":
    main()
