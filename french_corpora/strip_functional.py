import re
import fileinput

if __name__ == "__main__":
    for line in fileinput.input():
        print re.sub(r'-[^\s^\)]* |##[^\s^\)]*## ', ' ', line.strip())
