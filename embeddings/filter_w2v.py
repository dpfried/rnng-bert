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
    N_words = 0
    N_dims = None
    first_line = True

    lines_to_print = []

    for line in fileinput.input():
        if first_line:
            first_line = False
            N_dims = int(line.strip().split()[1])

        word = line[:line.index(' ')]
        if word in vocab:
            N_words += 1
            lines_to_print.append(line.strip())

    print("%s %s" % (N_words, N_dims))
    for line in lines_to_print:
        print(line)
