import re
import fileinput

if __name__ == "__main__":
    for line in fileinput.input():
        line = line.strip()
        if line.startswith("( "):
            assert line[-1] == ')'
            line = line[2:-1]
        print re.sub(r'-[^\s^\)]* |##[^\s^\)]*## ', ' ', line)
