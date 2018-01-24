import collections.abc

class Node(object):
    @property
    def label(self):
        raise NotImplementedError

    @property
    def children(self):
        raise NotImplementedError

    @property
    def word(self):
        raise NotImplementedError

    def linearize(self):
        raise NotImplementedError

class InternalNode(Node):
    def __init__(self, label, children):
        assert isinstance(label, str)
        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, Node) for child in children)
        self._label = label
        self._children = children

    @property
    def label(self):
        return self._label

    @property
    def children(self):
        return self._children

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

class LeafNode(Node):
    def __init__(self, label, word):
        assert isinstance(label, str)
        assert isinstance(word, str)
        self._label = label
        self._word = word

    @property
    def label(self):
        return self._label

    @property
    def word(self):
        return self._word

    def linearize(self):
        return "({} {})".format(self.label, self.word)

def read_trees(string):
    tokens = string.replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)
    return trees

def load_trees(path, strip_top=True):
    with open(path) as infile:
        trees = read_trees(infile.read())

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "S1"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    return trees
