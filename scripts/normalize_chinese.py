# normalize to be closer to CTB 5.1
import unicodedata

TOKEN_MAPPING = [
    ("———", "―――"),
    ("——", "――"),
    ("—", "―"),
    ("·", "・"),
    ("｛", "{"),
    ("｝", "}"),
    ("［", "["),
    ("］", "]")
]

def proc_line(line):
    line = line.rstrip()
    for key, value in TOKEN_MAPPING:
        line = line.replace(key, value)
    return line
        #line = unicodedata.normalize("NFKD", line)

if __name__ == "__main__":
    import fileinput
    for line in fileinput.input():
        print(proc_line(line))
