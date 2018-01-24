from os.path import join, basename
from lxml import etree
import sys
import codecs
from io import StringIO
from trees import read_trees

def separate_sentences(parse_string):
    spaced = parse_string.replace(")", " ) ").replace("(", " (")
    sents = []
    paren_count = 0
    sent = []
    for tok in spaced.split():
        if tok.startswith('('):
            paren_count += 1
        elif tok.startswith(')'):
            assert(tok == ')')
            paren_count -= 1
        sent.append(tok)
        if paren_count == 0:
            assert(sent[-1] == ')')
            if sent[0] == '(':
                assert(sent[-2] == ')')
                sent = sent[1:-1]
            sents.append(sent)
            sent = []
    return sents

tags = set()

def linearize(text):
    try:
        trees = read_trees(text)
        return [tree.linearize() for tree in trees]
    except AssertionError as e:
        print("malformed tree")
        return []
    except IndexError as e:
        print(e)
        return []

def parse_file(fname):
    if fname.endswith("bc"):
        with codecs.open(fname, 'r', 'utf-8') as f:
            return linearize(u' '.join(f.readlines()))
    sentences = []
    with codecs.open(fname, 'r', 'utf-8') as f:
        parser = etree.HTMLParser(
            recover=True,
            encoding='utf-8',
        )
        tree = etree.parse(
            StringIO(u'<body>%s</body>' % u' '.join(f)),
            parser
        )
    for elem in tree.iter():
        tag_lower = elem.tag.lower()
        tags.add(tag_lower)
        # if tag_lower not in ['text', 'turn', 's', 'seg', 'segment']:
        #     continue
        if not elem.text or not elem.text.strip() or not elem.text.strip().startswith('('):
            continue
        #sentences.append((elem.get("ID"), elem.text))
        sentences.extend(linearize(elem.text))
        # for tree in read_trees(elem.text):
        #     sentences.append(tree.linearize())
        # elem.clear()
        # while elem.getprevious() is not None:
        #     del elem.getparent()[0]
    # del context
    return sentences

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", nargs='+')
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(args.output_dir)

    N = len(args.input_files)
    for i, fname in enumerate(args.input_files):
        sys.stderr.write("\r%d / %d (%s)" % (i, N, fname))
        sentences = parse_file(fname)
        output_fname = join(args.output_dir, basename(fname) + ".bracketed")
        with codecs.open(output_fname, 'w', 'utf-8') as f:
            for sentence in sentences:
                f.write(u"%s\n" % sentence)
    sys.stderr.write("\n")

    print(tags)
