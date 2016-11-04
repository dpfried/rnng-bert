def proc_line(line):
    stripped = line.strip()
    assert(stripped.startswith("(TOP") and stripped.endswith(")"))
    return stripped[5:-1]

def proc_file(fname):
    with open(fname) as fin:
        with open(fname+".stripped", "w") as fout:
            for line in fin:
                fout.write(proc_line(line))
                fout.write("\n")


if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
