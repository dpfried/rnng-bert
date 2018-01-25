import re
import fileinput
from bllipparser import Tree

if __name__ == "__main__":
    for line in fileinput.input():
        line = line.strip()
        if line.startswith("( "):
            assert line[-1] == ')'
            line = line[2:-1]
        # this causes problems with -NONE- label, use bllipparser instead (which also will remove -NONE-)
        #line = re.sub(r'-[^\s^\)]* |##[^\s^\)]*## ', ' ', line)
        line = re.sub(r'##[^\s^\)]*## ', ' ', line)
        tree = Tree("(S1 " + line + ")")
        for subtree in tree.all_subtrees():
            subtree.label_suffix = ''
        linearized = str(tree)
        assert linearized.startswith("(S1 ") and linearized.endswith(")")
        print(linearized[4:-1])
