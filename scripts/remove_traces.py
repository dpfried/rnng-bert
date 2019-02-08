import bllipparser

legal_top = "S1"
legal_prefix = "({} ".format(legal_top)
legal_suffix = ")"

def read_tree(line):
    line = line.strip()
    added_top = False
    if not line.startswith(legal_prefix):
        added_top = True
        line = "{}{}{}".format(legal_prefix, line, legal_suffix)
    processed = bllipparser.Tree(line)
    str_rep = str(processed)
    if added_top:
        assert str_rep.startswith(legal_prefix)
        str_rep = str_rep[len(legal_prefix):]
        assert str_rep.endswith(legal_suffix)
        str_rep = str_rep[:-len(legal_suffix)]
    return str_rep

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(read_tree(line))
