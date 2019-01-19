import matplotlib.pyplot as plt
import re

def parse_file(filename):
    fscores = []
    with open(filename) as f:
        for line in f:
            match = re.match(r".*\*\*dev.* f1: ([0-9.]+).*", line)
            if match:
                fscores.append(float(match.group(1)))
    return fscores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--start-at", type=int)
    parser.add_argument("--end-at", type=int)
    args = parser.parse_args()
    by_name = {}
    for fname in args.files:
        series = parse_file(fname)
        if args.end_at:
            series = series[:args.end_at]
        if args.start_at:
            series = series[args.start_at:]
        by_name[fname] = series

    for name, series in sorted(by_name.items()):
        plt.plot(series, label=name)
    plt.legend(loc='lower right')
    plt.show()

