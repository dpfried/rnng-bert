def proc_line(line, symbol, must_have):
    stripped = line.strip()
    removed = False
    front = "({} ".format(symbol)
    if stripped.startswith(front):
        assert stripped.endswith(")")
        stripped = stripped[len(front):-1]
        removed = True
    if must_have:
        assert removed
    return stripped

if __name__ == "__main__":
    import fileinput
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="S1", required=True)
    parser.add_argument("--must_have", action="store_true")

    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = parser.parse_args()

    for line in fileinput.input(files=args.files if len(args.files) > 0 else ('-', )):
        print(proc_line(line, args.symbol, args.must_have))
