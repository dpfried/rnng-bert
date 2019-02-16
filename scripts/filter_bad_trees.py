import bllipparser
from remove_traces import read_tree
import sys

legal_top = "S1"
legal_prefix = "({} ".format(legal_top)
legal_suffix = ")"

if __name__ == "__main__":
    import fileinput
    for i, line in enumerate(fileinput.input()):
        try:
            line = line.rstrip()
            if line.count('(') != line.count(')'):
                sys.stderr.write("skipping line {} due to paren mismatch\n".format(i))
                sys.stderr.write("{}\n".format(line))
                continue
            if line.count("(") <= 1 or line.count(")") <= 1:
                sys.stderr.write("skipping line {} due to too few parens\n".format(i))
                sys.stderr.write("{}\n".format(line))
                continue
            if '\ue5f1' in line or '\ufffd' in line:
                sys.stderr.write("skipping line {} due to unicode special character\n".format(i))
                sys.stderr.write("{}\n".format(line))
                continue
            line = line.replace("( (", "(XX (")
            tree_str = read_tree(line + "\n")
            print(tree_str)
        except Exception as e:
            sys.stderr.write("skipping line {} due to error:\n".format(i))
            sys.stderr.write("{}\n".format(line))
            sys.stderr.write("{}\n".format(e))
            continue
