import re
import fileinput
import argparse

#!/usr/bin/env python

from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_from', nargs='*')
    parser.add_argument('--map_to', required=True)
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = parser.parse_args()

    for line in fileinput.input(files=args.files if len(args.files) > 0 else ('-', )):
        line = line.rstrip()
        for sym in args.map_from:
            line = line.replace("({} ".format(sym), "({} ".format(args.map_to))
            line = line.replace("({}-".format(sym), "({}-".format(args.map_to))
        print(line)
