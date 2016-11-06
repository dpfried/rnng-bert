from __future__ import print_function
import nltk

if __name__ == "__main__":
    vocab = set()
    fnames = ['../corpora/train.stripped',
             '../corpora/test.stripped',
             '../corpora/dev.stripped']
    for fname in fnames:
        with open(fname) as f:
            for line in f:
                tree = nltk.tree.Tree.fromstring(line)
                for word in tree.leaves():
                    vocab.add(word.lower())

    import fileinput
    for line in fileinput.input():
        word = line[:line.index(' ')]
        if word in vocab:
            print(line.strip())
