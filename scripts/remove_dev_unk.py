import sys
import fileinput
from strip_functional import PhraseTree

def remove_dev_unk(gold_line, sys_line):
    gold_tree = PhraseTree.parse(gold_line)
    sys_tree = PhraseTree.parse(sys_line)
    assert len(gold_tree.sentence) == len(sys_tree.sentence)
    for i in range(len(gold_tree.sentence)):
        sys_tree.sentence[i] = gold_tree.sentence[i]
    return str(sys_tree)

def main():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments: the gold dev set and the output file dev set')
    gold_file = open(sys.argv[1], 'r')
    gold_lines = gold_file.readlines()
    gold_file.close()

    # use fileinput so we can pass "-" to read from stdin
    sys_lines = list(fileinput.input(files=[sys.argv[2]]))

    assert len(gold_lines) == len(sys_lines)
    for gold_line, sys_line in zip(gold_lines, sys_lines):
        output_string = remove_dev_unk(gold_line, sys_line)
        print(output_string.rstrip())

if __name__ == '__main__':
    main()
