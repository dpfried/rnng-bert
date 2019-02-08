import sys
import get_dictionary
from get_oracle import unkify, get_nonterminal
from get_dictionary import is_next_open_bracket, get_between_brackets

# start_idx = open bracket
#def skip_terminals(line, start_idx):
#    line_end_idx = len(line) - 1
#    for i in range(start_idx + 1, line_end_idx):
#        if line[i] == ')':
#            assert line[i + 1] == ' '
#            return (i + 2)
#    raise IndexError('No close bracket found in a terminal')

def get_tags_tokens_lowercase(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]

def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                output_actions.append('SHIFT')
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
             output_actions.append('REDUCE')
             if i == max_idx:
                 break
             i += 1
             while line_strip[i] != ')' and line_strip[i] != '(':
                 i += 1
    assert i == max_idx
    return output_actions

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dictionary_file")
    parser.add_argument("corpus_file")
    parser.add_argument("--no_morph_aware_unking", action='store_true')
    args = parser.parse_args()

    # train_file = open(sys.argv[1], 'r')
    # # lines = train_file.readlines()
    # words_list = set(get_dictionary.get_dict(train_file) )
    # train_file.close()
    with open(args.dictionary_file, 'r') as f:
        words_list = set(line.strip() for line in f)

    with open(args.corpus_file, 'r') as dev_file:
        line_ctr = 0
        # get the oracle for the train file
        for i, line in enumerate(dev_file):
            if i % 1000 == 0:
                sys.stderr.write("\rget oracle %d" % i)
            line_ctr += 1
            # assert that the parenthesis are balanced
            if line.count('(') != line.count(')'):
                raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr))
            # first line: the bracketed tree itself itself
            print('# ' + line.rstrip())
            tags, tokens, lowercase = get_tags_tokens_lowercase(line)
            assert len(tags) == len(tokens)
            assert len(tokens) == len(lowercase)
            #print ' '.join(tags)
            print(' '.join(tokens))
            #print ' '.join(lowercase)
            unkified = unkify(tokens, words_list, not args.no_morph_aware_unking)
            print(' '.join(unkified))
            output_actions = get_actions(line)
            for action in output_actions:
                print(action)
            print('')


    if __name__ == "__main__":
        main()
