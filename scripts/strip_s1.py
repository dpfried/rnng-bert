def proc_line(line):
    stripped = line.strip()
    assert(stripped.startswith("(S1") and stripped.endswith(")"))
    return stripped[4:-1].replace("#", "*HASH*")

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
