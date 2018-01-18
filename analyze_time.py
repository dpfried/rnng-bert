import re

def parse_file(filename):
    times = []
    with open(filename) as f:
        for line in f:
            match = re.match(r".*\[([0-9.]+)ms per instance\]", line)
            if match:
                times.append(float(match.group(1)))
    return times

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    by_name = {}
    for fname in args.files:
        print(fname)
        times = parse_file(fname)
        if times:
            mean = sum(times) / len(times)
        else:
            mean = 0
        print("%s\t%s" % (mean, len(times)))
