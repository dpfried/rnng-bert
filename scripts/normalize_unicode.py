import unicodedata
import sys

paren_rep = {
    "（": " --LRB-- ",
    "）": " --RRB-- ",
    "［": " --LSB-- ",
    "］": " --RSB-- ",
    "【": " --LLB-- ",
    "】": " --RLB-- ",
    "，": " --CMM-- ",
    "！": " --BNG-- ",
    "？": " --QST-- ",
    "；": " --SMC-- ",
    "：": " --CLN-- ",
    "－": " --DSH-- ",
    "︶": " --UPC-- ",
}

def proc_line(line):
    line = line.rstrip()
    open_count = line.count("(")
    close_count = line.count(")")
    for key, value in paren_rep.items():
        line = line.replace(key, value)
    line = unicodedata.normalize("NFKC", line)
    for key, value in paren_rep.items():
        line = line.replace(value, key)
    if line.count("(") != open_count:
        sys.stderr.write("open counts don't match for line {}\n".format(line))
    if line.count(")") != close_count:
        sys.stderr.write("close counts don't match for line {}\n".format(line))
    return line

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
