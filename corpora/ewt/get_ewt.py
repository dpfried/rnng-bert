import glob
import os

LINE_START = "( "
LINE_START_SUB = "(TOP "

CATEGORIES = ["answers", "email", "newsgroup", "reviews", "weblog"]
SPLITPOINT_BY_CATEGORY = {
    "answers": -1744,
    "email": 2450,
    "newsgroup": -1195,
    "reviews": -1906,
    "weblog": 1016,
}

def proc_line(line):
    line = line.rstrip()
    assert line.startswith(LINE_START)
    assert line.count("(") == line.count(")")
    line = line[len(LINE_START):]
    line = LINE_START_SUB + line
    assert line.count("(") == line.count(")")
    return line

def proc_file(fname):
    with open(fname) as f:
        return [proc_line(line) for line in f]

def proc_folder(folder, file_ids=None):
    if file_ids is None:
        fnames = glob.glob(os.path.join(folder, "*.tree"))
    else:
        fnames = [os.path.join(folder, "{}.xml.tree".format(file_id)) for file_id in file_ids]

    lines = []
    for fname in sorted(fnames, key=lambda x: x.lower()):
        lines += proc_file(fname)
    return lines

def read_file_ids(fname):
    def proc_line(line):
        return line.rstrip().replace(".txt", "").replace(".snipped", "")
    with open(fname) as f:
        return [proc_line(line) for line in f]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../eng_web_tbk/data")
    parser.add_argument("--output_root", default=".")
    args = parser.parse_args()

    for category in CATEGORIES:
        folder = os.path.join(args.data_root, category, "penntree")
        assert os.path.exists(folder)
        file_ids = read_file_ids(os.path.join(args.data_root, "..", "docs", "file.ids.{}".format(category)))
        lines = proc_folder(folder, file_ids)

        dev_lines = lines[:SPLITPOINT_BY_CATEGORY[category]]
        test_lines = lines[SPLITPOINT_BY_CATEGORY[category]:]

        with open(os.path.join(args.output_root, "{}.dev.original".format(category)), "w") as f:
            for line in dev_lines:
                f.write("{}\n".format(line))

        with open(os.path.join(args.output_root, "{}.test.original".format(category)), "w") as f:
            for line in test_lines:
                f.write("{}\n".format(line))
