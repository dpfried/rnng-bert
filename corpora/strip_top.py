def proc_line(line):
    stripped = line.strip()
    assert(stripped.startswith("(TOP") and stripped.endswith(")"))
    return stripped[5:-1].replace("#", "*HASH*")

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
