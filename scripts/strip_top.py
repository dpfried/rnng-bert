def proc_line(line):
    stripped = line.strip()
    
    if stripped.startswith("(TOP") and stripped.endswith(")"):
    #return stripped[5:-1].replace("#", "*HASH*")
        return stripped[5:-1]
    else:
        return stripped

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
