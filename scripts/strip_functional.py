import re
import fileinput
import argparse

#!/usr/bin/env python

from collections import defaultdict

def remove_symbol_functionals(symbol):
    if symbol[0] == '-' and symbol[-1] == '-':
        return symbol
    morph_split = symbol.split('##')
    morph_split[0] = morph_split[0].split('-')[0]
    morph_split[0] = morph_split[0].split('=')[0]
    return '##'.join(morph_split)

class PhraseTree(object):
# adapted from https://github.com/jhcross/span-parser/blob/master/src/phrase_tree.py

    puncs = [",", ".", ":", "``", "''", "PU"] ## (COLLINS.prm)


    def __init__(
        self, 
        symbol=None, 
        children=[],
        sentence=[],
        leaf=None,
    ):
        self.symbol = symbol        # label at top node
        self.children = children    # list of PhraseTree objects
        self.sentence = sentence
        self.leaf = leaf            # word at bottom level else None
        
        self._str = None

    def remove_nodes(self, symbol_list):
        children = []
        for child in self.children:
            if child.symbol in symbol_list:
                children.extend(child.children)
            else:
                children.append(child)
        assert self.symbol not in symbol_list
        return PhraseTree(self.symbol, 
                          [child.remove_nodes(symbol_list) for child in children],
                          self.sentence,
                          leaf=self.leaf)

    def __str__(self):
        if self._str is None:
            if len(self.children) != 0:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1], 
                    self.sentence[self.leaf][0],
                )
        return self._str
        
    def pretty(self, level=0, marker='  '):
        pad = marker * level

        if self.leaf is not None:
            leaf_string = '({} {})'.format(
                    self.symbol, 
                    self.sentence[self.leaf][0],
            )
            return pad + leaf_string

        else:
            result = pad + '(' + self.symbol
            for child in self.children:
                result += '\n' + child.pretty(level + 1)
            result += ')'
            return result


    @staticmethod
    def parse(line):
        """
        Loads a tree from a tree in PTB parenthetical format.
        """
        line += " "
        sentence = []
        _, t = PhraseTree._parse(line, 0, sentence)

        if t.symbol == 'TOP' and len(t.children) == 1:
            t = t.children[0]
        
        return t


    @staticmethod
    def _parse(line, index, sentence):

        "((...) (...) w/t (...)). returns pos and tree, and carries sent out."
                
        assert line[index] == '(', "Invalid tree string {} at {}".format(line, index)
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ')':
            if line[index] == '(':
                index, t = PhraseTree._parse(line, index, sentence)
                if t is not None:
                    children.append(t)
            else:
                if symbol is None:
                    # symbol is here!
                    rpos = min(line.find(' ', index), line.find(')', index))
                    # see above N.B. (find could return -1)
                
                    symbol = line[index:rpos] # (word, tag) string pair

                    index = rpos
                else:
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    if symbol != '-NONE-':
                        sentence.append((word, remove_symbol_functionals(symbol)))
                        leaf = len(sentence) - 1
                    index = rpos
                
            if line[index] == " ":
                index += 1

        assert line[index] == ')', "Invalid tree string %s at %d" % (line, index)

        if symbol == '-NONE-' or (children == [] and leaf is None):
            t = None
        else:
            t = PhraseTree(
                symbol=remove_symbol_functionals(symbol), 
                children=children, 
                sentence=sentence,
                leaf=leaf,
            )

        return (index + 1), t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove_symbols', nargs='*')
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = parser.parse_args()

    remove_symbols = set(args.remove_symbols)

    for line in fileinput.input(files=args.files if len(args.files) > 0 else ('-', )):
        line = line.strip()
        if line.startswith("( "):
            assert line[-1] == ')'
            line = line[2:-1]
        # line = re.sub('\)', ') ', line)

        # #line = re.sub(r'-[^\s^\)]* |##[^\s^\)]*## ', ' ', line)
        # #line = re.sub(r'##[^\s^\)]*## ', ' ', line)
        # tokens = line.split(' ')
        # proc_tokens = []
        # last_was_none = False
        # for tok in tokens:
        #     if last_was_none:
        #         # follows '(-NONE-', should just be a terminal that looks like *op*, drop it
        #         assert not '(' in tok
        #         assert '*' in tok
        #         assert tok.count(')') == 1 and tok[-1] == ')'
        #         last_was_none = False
        #         continue

        #     if tok.startswith('('):
        #         if '-NONE-' in tok:
        #             last_was_none = True
        #             continue
        #         # remove functional tags -- anything after a - but not preceeded by ##
        #         morph_split = tok.split('##')
        #         morph_split[0] = re.sub('-.*', '', morph_split[0])
        #         tok = '##'.join(morph_split)
        #     proc_tokens.append(tok)

        # linearized = ' '.join(proc_tokens)
        # linearized = re.sub('\) ', ')', linearized)
        # print(linearized)
        tree = PhraseTree.parse(line)
        if args.remove_symbols:
            tree = tree.remove_nodes(remove_symbols)
        print(tree)
