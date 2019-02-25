import sys
import fileinput

def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')

def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)
    return ''.join(output)

def get_tags_tokens_lowercase(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]

def remove_dev_unk(gold_line, sys_line):
    gold_tags, gold_tokens, gold_lowercase = get_tags_tokens_lowercase(gold_line)
    sys_tags, sys_tokens, sys_lowercase = get_tags_tokens_lowercase(sys_line)
    assert len(gold_tokens) == len(gold_tags)
    assert len(gold_tokens) == len(gold_lowercase)
    assert len(gold_tokens) == len(sys_tokens)
    assert len(sys_tokens) == len(sys_tags)
    assert len(sys_tags) == len(sys_lowercase)
    output_string = sys_line
    for gold_token, gold_tag, sys_tag, sys_token in zip(gold_tokens, gold_tags, sys_tags, sys_tokens):
        output_string = output_string.replace('({} {})'.format(sys_tag, sys_token), '({} {})'.format(gold_tag, gold_token), 1)
    return output_string

def main():
    if len(sys.argv) != 3:
        raise NotImplementedError('Program only takes two arguments: the gold dev set and the output file dev set')
    gold_file = open(sys.argv[1], 'r')
    gold_lines = gold_file.readlines()
    gold_file.close()

    # use fileinput so we can pass "-" to read from stdin
    sys_lines = list(fileinput.input(files=[sys.argv[2]]))

    assert len(gold_lines) == len(sys_lines)
    for gold_line, sys_line in zip(gold_lines, sys_lines):
        output_string = remove_dev_unk(gold_line, sys_line)
        print(output_string.rstrip())

if __name__ == '__main__':
    main()
